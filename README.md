# SAQ
The official implementation of ICME 2024 paper "No-Reference Segmentation Annotation Quality Assessment".

The paper is selected for Oral Representation and Best Paper Candidate.

## Requirement
- Numpy
- Opencv  

## Useage
### Assessment for one pair:
```
python main.py  --img ./img_folder/a.jpg  --gt ./gt_folder/a.png
```
### Assessment for folders :
the image suffix can be jpg, png, bmp;  the gt suffix mush be png.
```
python main.py  --img ./img_folder  --gt ./gt_folder
```
### Choose backend  :
we support pure python, pytorch, and jittor, the default is python-cpu
```
--backend [python-cpu, pytorch-cpu, pytorch-gpu, jittor-cpu, jittor-gpu]
```
## Large-Scale Assessment for various datasets (updating)
See [this pdf file](https://github.com/frazerlin/SAQ/benchmark.pdf)

## Contact
If you have any questions, feel free to contact me via: `frazer.linzheng(at)gmail.com`.  
Welcome to visit [the project page](https://github.com/frazerlin/SAQ) or [my home page](https://www.lin-zheng.com/).

## License
The source code is free for research and education use only. Any commercial use should get formal permission first.
