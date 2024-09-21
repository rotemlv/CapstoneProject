import sys
from glob import glob
# Albumentations for augmentations
import albumentations as A
# visualization
import cv2
import matplotlib.pyplot as plt
import ml_collections
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib.patches import Rectangle
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader
# amd
import torch_directml

# GUI
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QHBoxLayout, \
    QMessageBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# model
from my_model import VisionTransformer, CONFIGS

# model

# CONSTANTS
TEST_BATCH_SIZE = 4
TRAINED_WEIGHTS_PATH= "my_weights.bin"
PATH_IMAGES = 'D:/kaggle/input/uwmgi-25d-stride2-dataset/images/images/*'
PATH_MASKS_CSV = 'D:/kaggle/input/uwmgi-mask-dataset/train.csv'


dml = torch_directml.device(torch_directml.default_device())

# config
class CFG:
    seed          = 101
    debug         = False # set debug=False for Full Training
    exp_name      = '2.5D'
    comment       = 'BifTransNet-MultiScale-244x244-ep=1'
    model_name    = 'MSBifTransNet'
    backbone      = 'ResNetV50'
    train_bs      = 16            # try 16 first
    valid_bs      = train_bs*2
    img_size      = [224, 224]
    epochs        = 8            # 7 FOR KAGGLE ONLY!!! start with 50 epochs (upgrade to 100 later)
    lr            = 4e-3  # modded from 2e-3
    scheduler     = 'CosineAnnealingLR'
    min_lr        = 5e-7  # modded from 1e-6
    T_max         = int(30000/train_bs*max(epochs, 20))+50
    T_0           = 25
    warmup_epochs = 0
    wd            = 1e-4  # modified from 1e-6 - attempt to avoid possible overfitting?
    n_accumulate  = max(1, 64//train_bs)  # modified to 64?
    n_fold        = 5
    folds         = [0,]  # for now only one fold
    num_classes   = 3
    device        = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device        = dml  # remove line if not using AMD GPU

def build_model():
    config_vit = CONFIGS['R50-ViT-B_16']
    config_vit.n_classes = CFG.num_classes
    config_vit.n_skip = 3
    config_vit.patches.grid = (
        int(224 / 16), int(224 / 16))
    model = VisionTransformer(config_vit, img_size=224, num_classes=CFG.num_classes)
    model.to(CFG.device)
    return model

def load_model(path):
    model = build_model()
    model.load_state_dict(torch.load(path, map_location=torch.device("cpu")))

    # amd
    model = model.to(dml)

    model.eval()
    return model

# load dataset:
path_df = pd.DataFrame(glob(PATH_IMAGES), columns=['image_path'])
path_df['mask_path'] = path_df.image_path.str.replace('image','mask')
path_df['id'] = path_df.image_path.map(lambda x: x.split('\\')[-1].replace('.npy',''))
path_df.head()

df = pd.read_csv(PATH_MASKS_CSV)
df['segmentation'] = df.segmentation.fillna('')
df['rle_len'] = df.segmentation.map(len) # length of each rle mask

df2 = df.groupby(['id'])['segmentation'].agg(list).to_frame().reset_index() # rle list of each id
df2 = df2.merge(df.groupby(['id'])['rle_len'].agg(sum).to_frame().reset_index()) # total length of all rles of each id

df = df.drop(columns=['segmentation', 'class', 'rle_len'])
df = df.groupby(['id']).head(1).reset_index(drop=True)
df = df.merge(df2, on=['id'])
df['empty'] = (df.rle_len==0) # empty masks

df = df.drop(columns=['image_path','mask_path'])
df = df.merge(path_df, on=['id'])
df.head()

fault1 = 'case7_day0'
fault2 = 'case81_day30'
df = df[~df['id'].str.contains(fault1) & ~df['id'].str.contains(fault2)].reset_index(drop=True)
df.head()

skf = StratifiedGroupKFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)
for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['empty'], groups = df["case"])):
    df.loc[val_idx, 'fold'] = fold
data_transforms = {
    "train": A.Compose([
        A.Resize(*CFG.img_size, interpolation=cv2.INTER_NEAREST),
        A.HorizontalFlip(p=0.5),
        #         A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.05, rotate_limit=10, p=0.5),
        A.OneOf([
            A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
            # #             A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=1.0),
            A.ElasticTransform(alpha=1, sigma=50, p=1.0)  # deleted "alpha_affine" for Colab
        ], p=0.25),
        A.CoarseDropout(max_holes=8, max_height=CFG.img_size[0] // 20, max_width=CFG.img_size[1] // 20,
                        min_holes=5, fill_value=0, mask_fill_value=0, p=0.5),
    ], p=1.0),

    "valid": A.Compose([
        A.Resize(*CFG.img_size, interpolation=cv2.INTER_NEAREST),
    ], p=1.0)
}


def load_img(path):
    img = np.load(path)
    img = img.astype('float32')  # original is uint16
    mx = np.max(img)
    if mx:
        img /= mx  # scale image to [0, 1]
    return img


def load_msk(path):
    msk = np.load(path)
    msk = msk.astype('float32')
    msk /= 255.0
    return msk


def show_img(img, mask=None):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    #     img = clahe.apply(img)
    #     plt.figure(figsize=(10,10))
    plt.imshow(img, cmap='bone')

    if mask is not None:
        # plt.imshow(np.ma.masked_where(mask!=1, mask), alpha=0.5, cmap='autumn')
        plt.imshow(mask, alpha=0.5)
        handles = [Rectangle((0, 0), 1, 1, color=_c) for _c in
                   [(0.667, 0.0, 0.0), (0.0, 0.667, 0.0), (0.0, 0.0, 0.667)]]
        labels = ["Large Bowel", "Small Bowel", "Stomach"]
        plt.legend(handles, labels)
    plt.axis('off')

class BuildDataset(torch.utils.data.Dataset):
    def __init__(self, df, label=True, transforms=None):
        self.df = df
        self.label = label
        self.img_paths = df['image_path'].tolist()
        self.msk_paths = df['mask_path'].tolist()
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = []
        img = load_img(img_path)

        if self.label:
            msk_path = self.msk_paths[index]
            msk = load_msk(msk_path)
            if self.transforms:
                data = self.transforms(image=img, mask=msk)
                img = data['image']
                msk = data['mask']
            img = np.transpose(img, (2, 0, 1))
            msk = np.transpose(msk, (2, 0, 1))
            return torch.tensor(img), torch.tensor(msk)
        else:
            if self.transforms:
                data = self.transforms(image=img)
                img = data['image']
            img = np.transpose(img, (2, 0, 1))
            return torch.tensor(img)

def prepare_loaders(fold, debug=False):
    train_df = df.query("fold!=@fold").reset_index(drop=True)
    valid_df = df.query("fold==@fold").reset_index(drop=True)
    if debug:
        train_df = train_df.head(32 * 5).query("empty==0")
        valid_df = valid_df.head(32 * 3).query("empty==0")
    train_dataset = BuildDataset(train_df, transforms=data_transforms['train'])
    valid_dataset = BuildDataset(valid_df, transforms=data_transforms['valid'])

    train_loader = DataLoader(train_dataset, batch_size=CFG.train_bs if not debug else 20,
                              num_workers=4, shuffle=True, pin_memory=True, drop_last=False)
    valid_loader = DataLoader(valid_dataset, batch_size=CFG.valid_bs if not debug else 20,
                              num_workers=4, shuffle=False, pin_memory=True)

    return train_loader, valid_loader


# train_loader, valid_loader = prepare_loaders(fold=0, debug=True)

def plot_batch(imgs, msks, size=3, do_show=True, title=None):
    plt.figure(figsize=(size * size, size))
    for idx in range(size):
        plt.subplot(1, size, idx + 1)
        img = imgs[idx,].permute((1, 2, 0)).numpy() * 255.0
        img = img.astype('uint8')
        msk = msks[idx,].permute((1, 2, 0)).numpy() * 255.0
        show_img(img, msk)
    plt.tight_layout()
    if title is not None:
        plt.suptitle(title, fontsize="large")

        # Set the window title
        plt.get_current_fig_manager().set_window_title(title)

    if do_show:
        plt.show()


class SegmentationGUI(QWidget):
    def __init__(self):
        super().__init__()

        self.curr_img = None
        self.curr_gt = None
        self.curr_preds = None
        self.view_mode = 'default'  # New attribute to track the view mode

        self.model = load_model(TRAINED_WEIGHTS_PATH)
        self.model.eval()

        self.test_dataset = BuildDataset(df.query("empty==0").sample(frac=1.0), label=True,
                                         transforms=data_transforms['valid'])
        self.test_loader = DataLoader(self.test_dataset, batch_size=1,
                                      num_workers=0, shuffle=False, pin_memory=True)

        self.current_index = 0
        self.total_images = len(self.test_loader)

        self.initUI()

    def initUI(self):
        self.setGeometry(300, 300, 1200, 600)
        self.setWindowTitle('Segmentation GUI - UWMGI')

        layout = QVBoxLayout()

        # Create a horizontal layout for buttons
        self.button_layout = QHBoxLayout()

        # Load new weights button
        self.load_weights_button = QPushButton("Load new weights")
        self.load_weights_button.clicked.connect(self.load_new_weights)
        self.button_layout.addWidget(self.load_weights_button)

        # Segment next image button
        self.next_button = QPushButton("Start Segmentation")
        self.next_button.clicked.connect(self.show_next_image)
        self.button_layout.addWidget(self.next_button)

        # Add button layout to main layout
        layout.addLayout(self.button_layout)

        # Add the new button
        self.view_button = QPushButton("View Over Image")
        self.view_button.clicked.connect(self.toggle_view_mode)
        self.button_layout.addWidget(self.view_button)

        self.figure = Figure(figsize=(10, 5))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        self.iter_loader = iter(self.test_loader)
        self.setLayout(layout)

    def toggle_view_mode(self):
        if self.view_mode == 'default':
            self.view_mode = 'over_image'
            self.view_button.setText("Default View")
        else:
            self.view_mode = 'default'
            self.view_button.setText("View Over Image")

        # Trigger segmentation again to update the view
        if self.curr_img is not None:
            self.plot_results(self.curr_img, self.curr_gt,  self.curr_preds)

    def load_new_weights(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Weights File", "", "All Files (*.*)")
        if file_path:
            try:
                self.model = load_model(file_path)
                self.model.eval()

                # Reset the test loader iterator
                self.reset_test_loader()

                # Show a success message box
                msg_box = QMessageBox()
                msg_box.setIcon(QMessageBox.Information)
                msg_box.setText("New weights loaded successfully!")
                msg_box.setInformativeText(f"Weights file: {file_path}")
                msg_box.setWindowTitle("Weights Loaded")
                msg_box.exec_()
                self.next_button.setText("Start Segmentation")

            except Exception as e:
                # Show an error message box
                msg_box = QMessageBox()
                msg_box.setIcon(QMessageBox.Critical)
                msg_box.setText("Error loading new weights")
                msg_box.setDetailedText(str(e))
                msg_box.setWindowTitle("Error Loading Weights")
                msg_box.exec__()
                print(f"{e}", file=sys.stderr)

    def open_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Image File", "", "Image Files (*.png *.jpg *.jpeg)")
        if file_path:
            self.file_path = file_path
            self.load_and_segment_image()

    def load_and_segment_image(self):
        if hasattr(self, 'file_path'):
            try:
                img = cv2.imread(self.file_path, cv2.IMREAD_UNCHANGED)
                if img is None:
                    raise ValueError(f"Failed to read image from {self.file_path}")

                # Print image shape for debugging
                print(f"Image shape: {img.shape}")

                # Normalize pixel values if necessary
                if img.dtype == np.uint16:
                    mx = np.max(img)
                    if mx:
                        img = img / mx

                # Resize to 224x224
                img = cv2.resize(img, (224, 224))

                # Ensure the image is in CHW format
                img = img[np.newaxis, :, :]

                # Convert to float tensor and reshape
                img_tensor = torch.FloatTensor(img).permute(0, 3, 1, 2)

                with torch.no_grad():
                    pred = self.model(img_tensor)
                    pred = ((nn.Sigmoid()(pred) > 0.5).float())

                self.plot_results_from_file(img_tensor, pred)

            except Exception as e:
                print(f"Error in load_and_segment_image: {str(e)}")
        else:
            print("No file path set.")

    def plot_results_from_file(self, img_tensor, pred):
        self.figure.clear()
        ax1 = self.figure.add_subplot(121)
        ax2 = self.figure.add_subplot(122)

        # Plot input image
        img = img_tensor.squeeze(0).permute(1, 2, 0).numpy()
        ax1.imshow(img)
        ax1.axis('off')
        ax1.set_title('Input Image')

        # Plot prediction
        pred = pred.squeeze(0).permute(1, 2, 0).numpy()
        ax2.imshow(pred, cmap='viridis')
        ax2.axis('off')
        ax2.set_title('Prediction')

        # Add legend to the rightmost subplot
        handles = [Rectangle((0, 0), 1, 1, color=_c) for _c in [(0.667, 0.0, 0.0), (0.0, 0.667, 0.0), (0.0, 0.0, 0.667)]]
        labels = ["Large Bowel", "Small Bowel", "Stomach"]
        ax2.legend(handles, labels, loc='upper center', bbox_to_anchor=(1.05, 1))

        self.canvas.draw()

    def plot_results(self, imgs, gt_msks, preds):
        self.figure.clear()
        if self.view_mode == 'default':
            ax1 = self.figure.add_subplot(221)
            ax2 = self.figure.add_subplot(222)
            ax3 = self.figure.add_subplot(223)
            ax4 = self.figure.add_subplot(224)
        elif self.view_mode == 'over_image':
            ax1 = self.figure.add_subplot(121)
            ax2 = self.figure.add_subplot(122)

        img = imgs[0].permute((1, 2, 0)).numpy() * 255.0
        img = img.astype('uint8')

        pred = preds[0].permute((1, 2, 0)).numpy()
        gt_msk = gt_msks[0].permute((1, 2, 0)).numpy()

        if self.view_mode == 'default':
            ax1.imshow(img, cmap='bone')
            ax1.axis('off')
            ax1.set_title('Input Image')

            ax2.imshow(gt_msk, cmap='viridis', vmin=0, vmax=1)
            ax2.axis('off')
            ax2.set_title('Ground Truth Mask')

            # Add legend for ground truth mask
            handles = [Rectangle((0, 0), 1, 1, color=_c) for _c in
                       [(0.667, 0.0, 0.0), (0.0, 0.667, 0.0), (0.0, 0.0, 0.667)]]
            labels = ["Large Bowel", "Small Bowel", "Stomach"]
            ax2.legend(handles, labels, loc='upper left', bbox_to_anchor=(0, 1),
                       fontsize=7, columnspacing=0.3, handletextpad=0.2)

            ax3.imshow(pred, cmap='viridis', vmin=0, vmax=1)
            ax3.axis('off')
            ax3.set_title('Predicted Mask')

            # Add legend for predicted mask
            handles = [Rectangle((0, 0), 1, 1, color=_c) for _c in
                       [(0.667, 0.0, 0.0), (0.0, 0.667, 0.0), (0.0, 0.0, 0.667)]]
            labels = ["Large Bowel", "Small Bowel", "Stomach"]
            ax3.legend(handles, labels, loc='upper left', bbox_to_anchor=(0, 1),
                       fontsize=7, columnspacing=0.3, handletextpad=0.2)

            diff = np.abs(pred - gt_msk)
            ax4.imshow(diff, cmap='viridis', vmin=0, vmax=1)
            ax4.axis('off')
            ax4.set_title('Difference')

        elif self.view_mode == 'over_image':
            # Prediction overlay
            pred_overlay = np.zeros_like(img)
            pred_overlay[:, :, 0] = img[:, :, 0] * (1 - pred[:, :, 0]) + pred[:, :, 0] * 255
            pred_overlay[:, :, 1] = img[:, :, 1] * (1 - pred[:, :, 1]) + pred[:, :, 1] * 255
            pred_overlay[:, :, 2] = img[:, :, 2] * (1 - pred[:, :, 2]) + pred[:, :, 2] * 255
            pred_overlay = pred_overlay.astype(np.uint8)

            ax1.imshow(pred_overlay)
            ax1.axis('off')
            ax1.set_title('Prediction Overlay')

            # Ground truth overlay
            gt_overlay = np.zeros_like(img)
            gt_overlay[:, :, 0] = img[:, :, 0] * (1 - gt_msk[:, :, 0]) + gt_msk[:, :, 0] * 255
            gt_overlay[:, :, 1] = img[:, :, 1] * (1 - gt_msk[:, :, 1]) + gt_msk[:, :, 1] * 255
            gt_overlay[:, :, 2] = img[:, :, 2] * (1 - gt_msk[:, :, 2]) + gt_msk[:, :, 2] * 255
            gt_overlay = gt_overlay.astype(np.uint8)

            ax2.imshow(gt_overlay)
            ax2.axis('off')
            ax2.set_title('Ground Truth Overlay')

        self.canvas.draw()
        self.next_button.setText("Segment Next Image")

    def run_segmentation(self):
        try:
            imgs, gt_msks = next(self.iter_loader)
            self.current_index += 1
        except StopIteration:
            self.reset_test_loader()
            imgs, gt_msks = next(self.iter_loader)

        imgs = imgs.to(CFG.device, dtype=torch.float)

        with torch.no_grad():
            pred = self.model(imgs)
            pred = [(nn.Sigmoid()(pred) > 0.5).double()]
            imgs = imgs.cpu().detach()
            preds = torch.mean(torch.stack(pred, dim=0), dim=0).cpu().detach()
        self.curr_img = imgs
        self.curr_gt = gt_msks
        self.curr_preds = preds
        self.plot_results(imgs, gt_msks, preds)

    def reset_test_loader(self):
        self.iter_loader = iter(self.test_loader)
        self.current_index = 0

    def show_next_image(self):
        if self.current_index < self.total_images - 1:
            self.run_segmentation()
        else:
            print("End of dataset reached.")

def main():
    app = QApplication(sys.argv)
    gui = SegmentationGUI()
    gui.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()


def main():
    app = QApplication(sys.argv)
    gui = SegmentationGUI()
    gui.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
