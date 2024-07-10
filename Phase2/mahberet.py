
# IMPORTANT: RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES
# TO THE CORRECT LOCATION (/kaggle/input) IN YOUR NOTEBOOK,
# THEN FEEL FREE TO DELETE THIS CELL.
# NOTE: THIS NOTEBOOK ENVIRONMENT DIFFERS FROM KAGGLE'S PYTHON
# ENVIRONMENT SO THERE MAY BE MISSING LIBRARIES USED BY YOUR
# NOTEBOOK.

import os
import sys
from tempfile import NamedTemporaryFile
from urllib.request import urlopen
from urllib.parse import unquote, urlparse
from urllib.error import HTTPError
from zipfile import ZipFile
import tarfile
import shutil
# install this one!
from pyuac import main_requires_admin


@main_requires_admin
def main():
    CHUNK_SIZE = 40960
    DATA_SOURCE_MAPPING = 'uw-madison-gi-tract-image-segmentation:https%3A%2F%2Fstorage.googleapis.com%2Fkaggle-competitions-data%2Fkaggle-v2%2F27923%2F3495119%2Fbundle%2Farchive.zip%3FX-Goog-Algorithm%3DGOOG4-RSA-SHA256%26X-Goog-Credential%3Dgcp-kaggle-com%2540kaggle-161607.iam.gserviceaccount.com%252F20240710%252Fauto%252Fstorage%252Fgoog4_request%26X-Goog-Date%3D20240710T180034Z%26X-Goog-Expires%3D259200%26X-Goog-SignedHeaders%3Dhost%26X-Goog-Signature%3D14f0ae5a152a51a1662637eb3fd58cd8559ac513c04d0086b25203d405c8139d88919ffafcb4d77d14f6816b448ae368a1e6a6dc1f90f497fb037f24612a3da1f1c07886e5d8dac428bac806884f2c14f8c5f21746d2f0dbed2d62861a4748c95d5f0eb0f6cd94b93377e4f4aeec1c6e4475803f1821d9e575c49f55f28600af19239702aa17496de20f580010ff4512a5f33451cdfa5cafa65cca866f57662fc0764d11edff9dea091d38d459849397de207e0914a43ae35f040d02dbf59f96c1364f884e064fcf599e1aac96fbc7409069458a22afbc200bc9c435b9923d42aa1d513511a533ba1bc1d2b8333ab32c79ecf312940c90f4ad00e294cfdd1c72,pytorch-segmentation-models-lib:https%3A%2F%2Fstorage.googleapis.com%2Fkaggle-data-sets%2F1701116%2F2786089%2Fbundle%2Farchive.zip%3FX-Goog-Algorithm%3DGOOG4-RSA-SHA256%26X-Goog-Credential%3Dgcp-kaggle-com%2540kaggle-161607.iam.gserviceaccount.com%252F20240710%252Fauto%252Fstorage%252Fgoog4_request%26X-Goog-Date%3D20240710T180034Z%26X-Goog-Expires%3D259200%26X-Goog-SignedHeaders%3Dhost%26X-Goog-Signature%3D450828d385da34b48209fa5c45d28a54ab851a4110e8a719fc321ae104e4084acd72eaddc4585b6f82ab64ffdfb87e3da01098113b5724daf1e937ade109aa7c4074388f3824e2c8157a372618f15f69389173f3ea3be4e92570816d691a404debb188600db53cc2f3bcd33c70ba72971222c29db31bf1100405ef5953c0d159674f2a9f346f68287263de863da0886937e66988cb4299ca757cdeea80e788edfa7967546ac2d01f37baf064a158f6788b63e88e2dbebacd6d6f0e84a03eca0f937eeaf6a07b2e64b7a62c6073abd613bd460156913332e51fa76391262170146b36bab9f1c80dea39f3450cdabe2fdbc51a7e82420d3e95b47db2a826c536a4,uwmgi-mask-dataset:https%3A%2F%2Fstorage.googleapis.com%2Fkaggle-data-sets%2F2089184%2F3521209%2Fbundle%2Farchive.zip%3FX-Goog-Algorithm%3DGOOG4-RSA-SHA256%26X-Goog-Credential%3Dgcp-kaggle-com%2540kaggle-161607.iam.gserviceaccount.com%252F20240710%252Fauto%252Fstorage%252Fgoog4_request%26X-Goog-Date%3D20240710T180035Z%26X-Goog-Expires%3D259200%26X-Goog-SignedHeaders%3Dhost%26X-Goog-Signature%3D881f5fb7692c255031cb956856f6f5474737ded4d8adb230f0f1eff6367be145d51cab8fd259984c9040148509d6f12ac7e20e7f95875c36ee47109ad51321ff47cecd8ffe33c3493bb461d404cc3f7827357215002c0c7e82c1a0ddc1e460a54d1485a373d9c73e834e4cb6fee0d34fdcc9855e80bc747dd24a382712e25896ebe4e744e2593d86515322d2f50e1e58a40edeafdf7d33ee3270113b8fa8cc89b19583f81dc372bfd65dc7e25f9f0ee9730dd87030d1c05f30f16366ec2d34fbd8d62d40ab44848e7887343b8cc90b4335abdc4135ae8e71cdd3efb39f16f1a4aec2bc82dbb366b2c24ef1b7a518944191d84e7746632d6465dc26110b0bee5a,uwmgi-25d-stride2-dataset:https%3A%2F%2Fstorage.googleapis.com%2Fkaggle-data-sets%2F2142905%2F3566043%2Fbundle%2Farchive.zip%3FX-Goog-Algorithm%3DGOOG4-RSA-SHA256%26X-Goog-Credential%3Dgcp-kaggle-com%2540kaggle-161607.iam.gserviceaccount.com%252F20240710%252Fauto%252Fstorage%252Fgoog4_request%26X-Goog-Date%3D20240710T180035Z%26X-Goog-Expires%3D259200%26X-Goog-SignedHeaders%3Dhost%26X-Goog-Signature%3D738170c18954858d017faf879e70f7158d5ea1427ff17f29ee0f0fa6949cf1d4f7ceba44d39d387c13b79e4375813fe46f6a4ba24c9a4f0d914d4580236a56de8ccc21f2d359877190f95633baa1c109efa44b05ddd7083b3b576b841463362924ba308f27855bd4a4f129863a0b89dafb61d1ad722f1ee451ef042e4f0671c3fd1c6805001229314987bb2aa6ef0aeaa234fea18028ed69cbdb6c2eb5d7b2e918ae10ecbb9c9b70db4fb2394c24243a0309ba371b6595a8504b2197f5e461dcda48de7e0ee3808380ea672abc62fc1a0ed5240702bc49c697886c542218cc98b5dd61b3f553247774120933c389e4a8202b8b8958b575d1389e8baed5c325e3'

    KAGGLE_INPUT_PATH='/kaggle/input'
    KAGGLE_WORKING_PATH='/kaggle/working'
    KAGGLE_SYMLINK='kaggle'

    shutil.rmtree('/kaggle/input', ignore_errors=True)
    os.makedirs(KAGGLE_INPUT_PATH, 0o777, exist_ok=True)
    os.makedirs(KAGGLE_WORKING_PATH, 0o777, exist_ok=True)

    try:
      os.symlink(KAGGLE_INPUT_PATH, os.path.join("..", 'input'), target_is_directory=True)
    except FileExistsError:
      pass
    try:
      os.symlink(KAGGLE_WORKING_PATH, os.path.join("..", 'working'), target_is_directory=True)
    except FileExistsError:
      pass

    for data_source_mapping in DATA_SOURCE_MAPPING.split(','):
        directory, download_url_encoded = data_source_mapping.split(':')
        download_url = unquote(download_url_encoded)
        filename = urlparse(download_url).path
        destination_path = os.path.join(KAGGLE_INPUT_PATH, directory)
        try:
            with urlopen(download_url) as fileres, NamedTemporaryFile() as tfile:
                total_length = fileres.headers['content-length']
                print(f'Downloading {directory}, {total_length} bytes compressed')
                dl = 0
                data = fileres.read(CHUNK_SIZE)
                while len(data) > 0:
                    dl += len(data)
                    tfile.write(data)
                    done = int(50 * dl / int(total_length))
                    sys.stdout.write(f"\r[{'=' * done}{' ' * (50-done)}] {dl} bytes downloaded")
                    sys.stdout.flush()
                    data = fileres.read(CHUNK_SIZE)
                if filename.endswith('.zip'):
                  with ZipFile(tfile) as zfile:
                    zfile.extractall(destination_path)
                else:
                  with tarfile.open(tfile.name) as tarfile:
                    tarfile.extractall(destination_path)
                print(f'\nDownloaded and uncompressed: {directory}')
        except HTTPError as e:
            print(f'Failed to load (likely expired) {download_url} to path {destination_path}')
            continue
        except OSError as e:
            print(f'Failed to load {download_url} to path {destination_path}')
            continue

    print('Data source import complete.')


if __name__ == '__main__':
    main()