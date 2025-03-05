## Project Preview:
Our project focuses on advancing the field of gastrointestinal (GI) tract segmentation, aiming to
enhance the differentiation between healthy organs and tumor tissues for improved radiation
therapy planning. We are leveraging the transformative potential of deep learning, specifically by
integrating advanced attention mechanisms and a multi-scale context approach into a U-Net-
based model.
By incorporating attention mechanisms, our model will be able to prioritize the most relevant
features within the images, enabling a deeper understanding of the anatomical nuances. This
enhanced comprehension can help in accurately identifying tumor boundaries within the intricate
structures of the GI tract.
Furthermore, we plan to include the integration of a multi-scale context approach. This feature
allows our model to analyze images at various scales, capturing both the detailed characteristics
of both low-level and high-level semantic information. This perspective can help us provide a
thorough analysis of the tissue, and provide an accurate distinction between healthy and
malignant tissues.
Through these innovations, our project aims to develop a tool that surpasses current methods of
GI tract segmentation, setting a new benchmark in the field. The ultimate objective is to
contribute to the advancement of radiation therapy planning, leading to more effective treatments
and improved patient outcomes.

## Additional info:
Implementation ahead, components to be built (in folder "Stage2"):
-  :heavy_check_mark:  BiFusion block: fixed using the BiFTransNet paper's mathematical formulae
-  :heavy_check_mark:  TUP block
-  ✖️ Coarse-to-fine attention block (according to the 3D-TransUNet specs) - scrapped due to GPU memory cost.
-  :heavy_check_mark:  Multi-scale skip connections: either Seminar TransAttUnet strategy or imo MultiTrans can also be good)
-  :heavy_check_mark:  Data augmentation techniques
-  ✖️ Multi-level loss (as part of the training process) - tested and removed due to lower accuracy.
-  :heavy_check_mark:  Test hyperparameters
-  :heavy_check_mark:  Review results
-  :heavy_check_mark:  GUI wrapper
