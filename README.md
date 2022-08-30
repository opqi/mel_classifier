# Audio Classifier / Mel Spectogram Classifier

Сlassifies presence of a noise in an audio.

# Run:
+ train launch:
    $ ./run.py train/ --launch_type train --batch_size batch_size --epochs num_epochs
    --lr lr --channels channels --duration duration --device device --model model_path
    --model_arch model_arch --num_classes num_classes

    Try it out: ./run.py train/

+ val launch:
    $ ./run.py val/ --launch_type val --batch_size batch_size
    --channels channels --duration duration --device device --model model_path
    --model_arch model_arch --num_classes num_classes
    
    Try it out: ./run.py val/ --launch_type val --model pre_trained_resnet50.pth

+ test launch:
    $ ./run.py data_path --launch_type test --channels channels 
    --duration duration --device device --model model_path
    --model_arch model_arch --num_classes num_classes
    
    Try it out: $ ./run.py data_path --launch_type test --model pre_trained_resnet50.pth

Currently only 3 models supported:
+ ViT
+ Pretrained on ImageNet ResNet-50
+ DemoClassifier - 4 block Conv Net