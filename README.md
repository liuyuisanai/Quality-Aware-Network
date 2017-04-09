## Quality Aware Network

Codebase for (Partial) Quality Aware Network. Notice that QAN in ''Quality Aware Network for Set to Set Recognition'' is a 'single-part' version of P-QAN in ''Unsupervised Partial Quality Predictor for Large-Scale Person Re-identification'', so you can set part number to 1 if you wanna reproduce results of QAN.

**QAN** is used to handle set-to-set recognition problem. It can automatically learn the quality score for each sample in a set, and use the score to be the weight for synthesizing feature.

**P-QAN** is our subsequent work after QAN. It is used to handle video based person re-identification problem with awareness of partial quality. It can learn not only the quality of each frame but also the occlusion/blur/noise/etc. level in each part of a frame.

We use the [PRID 2011](https://lrs.icg.tugraz.at/datasets/prid/), [iLIDS-VID](www.eecs.qmul.ac.uk/.../downloads_qmul_iLIDS-VID_ReID_dataset.html) and [LPW] datasets for evaluation. It is a video-based task for person re-identification.

## Running code

1.Complile [CaffeMex\_v2](https://github.com/sciencefans/CaffeMex_v2) with matlab interface (see readme in CaffeMex\_v2).

2.Configure the path `CaffeMex_v2/matlab/+caffe` to the directory in this project.

3.Configure the parameters in `train_baseline/train_baseline.m`, `train_baseline/train_LPW.m` and `train_PQAN/train_LPW.m`, `train_PQAN/train_network_and_test.m`, including the path of prototxt and the relative param.

4.Running the scripts in the `generate_data` to generate your dataset split.

5.Modify the number of corresponding network classifications.
Running the `train_baseline/train_baseline.m` or `train_baseline/train_LPW.m` for baseline and `train_PQAN/train_LPW.m`, `train_PQAN/train_network_and_test.m` for PQAN.

## Q&A

Here we list some commonly asked questions we received from the public. Thanks for your engagement to make our work better!

- *Any strategies are used to make the quality scores effective?

 There are two main aspects that will affect the effectiveness of the quality scores. One is using a pre-trained model for imagenet task in order to make the network converge well. In the training stage, the use_global_stats is set to false in the Batchnorm layer; The other is the configuration of the parameter like learning rate, margin in the tripletloss layer, and you can adjust the parameters by observing the change of the scores in the training stage.

## Still having questions?
Feel free to drop us an email sharing your ideas.


## Citation
Please kindly cite our work in your publications if it helps your research:

    @inproceedings{liu_2017_qan,
      Author = {Liu, Yu and Junjie, Yan and Ouyang, Wanli},
      Title={Quality Aware Network for Set to Set Recognition},
	  Booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
	  Year = {2017}
    }
