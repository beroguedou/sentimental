# Sentimental

## Launch a training

First of all you should create a directory with the name "saved" when the training function will save the future trained model. 
```bash
cd sentimental
mkdir saved
```

And then you could run a training by running the following bash command:

```bash
python train.py --epochs 30 --batch 3 --workers 5 --stopping 5 --device 'cuda:0' --name "sentimental_camembert_model.pth"
```

## Serve a prediction
To serve prediction we made a basic flask application that is connected to a model trained (model should be in the directory "saved").
```bash
python sentimental_french_app.py
```
