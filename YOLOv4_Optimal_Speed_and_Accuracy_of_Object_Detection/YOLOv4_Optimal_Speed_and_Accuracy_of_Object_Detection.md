# [논문리뷰] YOLOv4: Optimal Speed and Accuracy of Object Detection

## 1. Introduction

대부분의 CNN 기반 object detector는 recommendation systems에 적용 가능하다. Real-time object detector에 사용하기 위해선 높은 accuracy와 FPS를 필요로 한다.

- **YOLOv4의 목적**
    - 생산 시스템에서 object detector의 빠른 작동 속도와 병렬 연산을 위한 최적화.

![image1](images/image1.png)

- **YOLOv4의 기여**
    - 1080Ti, 2080Ti를 사용하여 누구나 효과적이고 강력한 object detection model을 학습시킬 수 있다.
    - Detector 학습 도중 object detection 최신 기법인 bag-of-freebies와 bag-of-specials 기법의 영향을 검증하였다.
    - Single CPU 학습에 더 적합하도록 CBN(Cross-Iteration Batch Normalization), PAN(Path Aggregation Network), SAM(Spatial Attention Module)과 같은 최신 기법을 수정하였다.

## 2. Related work

### 2.1 Object detection models

![image2](images/image2.png)

- **Input:** Image, Patches, Image Pyramid
- **Backbones:**
    - **GPU platform**: VGG, ResNet, ResNeXt, DenseNet
    - **CPU platform**: SqueezeNet, MobileNet, ShuffleNet
- **Neck**: FPN, PAN, BiFPN, NAS-FPN
- **Head**:
    - **One-stage**:
        - **Anchor based**: YOLO, SSD, RetinaNet
        - **Anchor free**: CenterNet, CornerNet, FCOS
    - **Two-stage**: fast R-CNN, faster R-CNN, R-FCN, Libra R-CNN
    

### 2.2 Bag of freebies

**BOF(Bag of freebies)란**: inference 비용을 늘리지 않고 정확도를 향상시키는 방법

1. **Data augmentation**: 입력 이미지의 변동성을 증가시켜 object detection model이 다른 환경에서 얻은 이미지에 대하여 더 높은 robustness를 갖도록 한다
    - **Photometric distortions**: brightness, contrast, hue, saturation, noise of an image
    - **Geometric distortions**:  random scaling, cropping, flipping, rotating
    - Random ease, CutOut, hide-and-seek, grid mask
    - MixUp, CutMix
    - GAN
2. **Imbalance sampling**: 다양한 클래스사이의 data imbalance 문제를 해결
    - Focal Loss
    - **Label smoothing:** 모델이 더 높은 robustness를 갖도록 학습시 hard label를 soft label로 변환한다
3. **Objective function:** 기존의 Mean Square Error(MSE) 대신에 아래 loss function을 사용
    - **IoU loss:** 예측된 bbox와 ground truth bbox의 coverage를 고려
    - **GIoU loss:** coverage area와 object shape 및 orientation을 포함
    - **DIoU loss:** 추가적으로 object의 중심 거리를 고려
    - **CIoU loss:** overlapping area, center points 사이의 거리, aspect ratio를 동시에 고려

### 2.3 Bag of Specials

**BOS(Bag of specials)란**: inference 비용을 조금 높이면서 정확도를 크게 향상시키는 방법.

1. **Plugin modules**: 모델의 특정 속성을 강화시키는 방법(Enhancing certain attributes in a model)
    - **Enhancing receptive field:** SPP, ASPP, RFB
    - **Attention module:** Squeeze-and-Excitation(SE), Spatial Attention Module(SAM)
    - **Feature integration:** SFAM, ASFF, BiFPN
    - **Activation function:** Gradient vanish problem을 해결하기 위하여 ReLU 함수가 쓰였고, 이후 LReLU, PReLU, ReLU6, Scaled Exponential Linear Unit(SELU), Swish, hard-Swish, Mish가 제안되었다.
        - Swish, Mish: Continuously differentiable activation function
2. **Post-processing**: 모델의 예측 결과를 선별하는 방법(Screening model prediction results)
    - **NMS(Non-Maximum Suppression):** 동일한 object에 대하여 생성된 bbox중 high reponse를 가진 bbox만 남기고 필터링 하는 것(Filter those BBoxes that badly predict the same object, and only retain the candidate BBoxes with higher response)
    
    Anchor-free method에서는 captured image feature을 직접 참고하지 않기 때문에 post-processing이 더 이상 필요하지 않음
    

## 3. Methodology

### 3.1 Selection of architecture

![image3](images/image3.png)

**목적:**

- Input network의 해상도, the convolutional layer 개수, the parameter 개수, layer outputs 개수 사이에서 최적의 balance를 찾는 것
- Receptive field를 늘리기 위한 additional block과 서로 다른 detector level에 대한 서로 다른 parameter aggregation을 위한 최상의 method를 선택

**Architecture 선택 조건:**

- **Higher input network size** - for detecting multiple small-sized objects
- **More layers** - for a higher receptive filed to cover the increased size of input network
- **More parameters** - for greater capacity of a model to detect multiple objects of different sizes in a single image

**Finally:**

- **Backbone:** CSPDarknet53
- **Additional module:** SPP
- **Path-aggregation:** PANet
- **Head:** YOLOv3

### 3.2 Selection of BoF and BoS

- **Activations:** ReLU, leaky-ReLU, Swish, Mish
- **BBox regression loss:** MSE, IoU, GIoU, CIoU, DIoU
- **Data augmentation:** CutOut, MixUp, CutMix
- **Regularization method:** DropBlock
- **Normalization of the network activations by their mean and variance:** Batch Normalization (BN), Filter Response Normalization (FRN), Cross-Iteration Batch Normalization (CBN)
- **Skip-connections:** Residual connections, Weighted residual connections, Multi-input weighted residual connections, Cross stage partial connections (CSP)

### 3.3 Additional improvements

Single GPU에서 디자인한 detector가 더 잘 학습할 수 있도록 추가적인 설계를 도입

- 새로운 data augmentation 기법인 Mosaic과 Self-Adversarial Training (SAT)

![image4](images/image4.png)

- General algorithms을 적용하여 최적의 hyper-parameter를 선택
- 기존에 존재하는 기법을 수정하여 적용: modified SAM, modified PAN, Cross mini-Batch Normalization(CmBN)
    - **CmBN:** CBN의 수정된 버전으로 single batch 내에서 mini-batches 사이에 대한 통계를 수집
        
        ![image5](images/image5.png)
        
    
    - **Modified SAM:** Spatial-wise attention에서 point-wise attension으로 변경
    - **Modified PAN:** Shortcut connection을 concatenation으로 교체

![image6](images/image6.png)

### 3.4 YOLOv4

**YOLOv4 consists of:**

- **Backbone:** CSPDarknet53
- **Neck:** SPP, PAN
- **Head:** YOLOv3

**YOLOv4 uses:**

- **BoF for backbone:** CutMix and Mosaic data augmentation, DropBlock regularization, Class label smoothing
- **BoS for backbone:** Mish activation, Cross-stage partial connections (CSP), Multi-input weighted residual connections (MiWRC)
- **BoF for detector:** CIoU-loss, CmBN, DropBlock regularization, Mosaic data augmentation, Self-Adversarial Training (SAT), Eliminate grid sensitivity, Using multiple anchors for a single ground truth, Cosine annealing scheduler, Optimal hyper-parameters, Random training shapes
- **BoS for detector:** Mish activation, SPP-block, SAM-block, PAN path-aggregation block, DIoU-NMS

## 4. Experiments

### 4.2 Influence of different features on Classifier training

Classfier training 할 때 서로 다른 feature가 미치는 영향에 대하여 연구하였으며 각 feature들은 아래와 같다

- Class label smoothing
- different data augmentation techniques
- bilateral blurring
- MixUp, CutMix and Mosaic
- different activations (Leaky-ReLU, Swish, Mish)

![image7](images/image7.png)

결과적으로 classfier training 성능을 향상시키는 feature들은 아래와 같다

- CutMix and Mosaic for data augmentation
- Class label smoothing
- Mish activation

![image8](images/image8.png)

### 4.3 Influence of different features on Detector training

서로 다른 BoF-detector가 detector training accuracy에 주는 영향을 보기 위하여 추가적인 연구를 진행하였으며, FPS에 영향을 주지 않으면서 detector accuracy를 높히는 다양한 feature들에 대한 연구를 통하여 BoF list를 크게 확장함

- **S:** Eliminate grid sensitivity the equation $b_x=\sigma(t_x)+c_x, b_y=\sigma(t_y)+c_y$ where $c_x$ and $c_y$ are always whole numbers, is used in YOLOv3 for evaluating the object coordinates, therefore, extremely high $t_x$ absolute values are required for the $b_x$ value approaching the $c_x$ or $c_x + 1$ values.
- **M:** Mosaic data augmentation
- **IT:** IoU threshold - using multiple anchors for a single ground truth IoU (truth, anchor) > IoU threshold
- **GA:** Genetic algorithms - using genetic algorithms for selecting the optimal hyperparameters during network training on the first 10% of time periods
- **LS:** Class label smoothing
- **CBN:** CmBN - using Cross mini-Batch Normalization
- **CA:** Cosine annealing scheduler - altering the learning rate during sinusoid training
- **DM:** Dynamic mini-batch size - automatic increase of mini-batch size during small resolution training by using Random training shapes
- **OA:** Optimized Anchors - using the optimized anchors for training with the 512x512 network resolution
- **GIoU, CIoU, DIoU, MSE:** using different loss algorithms for bbox regression

![image9](images/image9.png)

BoS-detector 또한 detector training accuracy의 향상에 있어서 많은 영향을 주었으며 추가적인 연구를 진행하여 SPP, PAN, SAM을 사용하였을 때 가장 좋은 성능을 보임

![image10](images/image10.png)

### 4.4 Influence of different backbones and pretrained weightings on Detector training

![image11](images/image11.png)

다른 backbone model이 detector accuracy에 미치는 영향을 보기 위하여 추가로 연구를 진행하였고, 결과는 위 표와 같으며, best classification accuracy를 갖는 모델이 항상 best detector accuracy를 갖는 것이 아님을 발견

- CSPResNeXt-50 모델의 classification accuracy가 높은 반면, object detection에선 CSPDarknet53 모델이 더 높은 accuracy를 가짐
- CSPResNeXt50 classifier training 모델에 BoF와 Mish를 적용하여 classification accuracy를 향상시킨 반면, detector accuracy는 오히려 떨어짐. 이와 반대로 CSPDarknet53 classifier training 모델에 BoF와 Mish를 적용하였을 때는 classification accuracy와 detector accuracy 둘 다 상승함

결론적으로 CSPDarknet53 모델이 detector accuracy를 개선할 수 있는 능력이 더 뛰어남

### 4.5 Influence of different mini-batch size on Detector training

![image12](images/image12.png)

- BoF와 BoS training strategy를 추가한 후, mini-batch size는 detector performance에 영향을 거의 미치지 않음
- BoF와 BoS를 도입한 후 누구나 개인 GPU를 사용하여 완벽한 detector를 학습시킬 수 있음

## 5. Results

![image13](images/image13.png)

- YOLOv4 모델이 speed와 accuracy 모두 다른 SOTA 모델에 비하여 뛰어난 성능을 보임

## 6. Conclusions

- 뛰어난 FPS와 높은 정확도를 가진 SOTA 모델을 제안함
- 8-16 GB-VRAM을 가진 개인용 GPU에서 학습 가능한 모델
- One-stage anchor-based detector의 가능성을 입증함
- 수많은 feature들을 도입하여 classifier와 detector의 성능을 높일 수 있다는 것을 입증함
- 이러한 feature들은 이 후에 있을 연구와 개발에 쓰일 수 있음