# Dataset

For this project, we will be using DAIC-WOZ dataset from the USC Institute for Creative technologies.  

The dataset can be requested here: https://dcapswoz.ict.usc.edu/  

**To download the dataset upon approval**:  
*(ensure to use wsl)*

```python
wget -r -np -nH --cut-dirs=1 -R "index.html*" --user=YOUR_USERNAME --ask-password "<link_to_dataset>"
```

In the event of *interrupted download*:  

```python
wget -c -r -np -nH --cut-dirs=1 -A "*.zip,*.csv" -R "index.html*" --user=YOUR_USERNAME --ask-password "<link_to_dataset>"
```

## 1. Dataset Proprocessing

Within each .zip file in the dataset, there is an audio file with an accompanying transcript. So, we use that transcript to extract speech by the patient. Then we stitch the patient's audio together to form a filtered audio file.  

We resampled these audio files to 16 kHz, segment them into fixed size windows (default 8s) and use these segments to create log-mel spectrograms that are used as input later.  

Since the dataset already has train-validation-test split, we shall use their split for simplicity.