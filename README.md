# Object Detection Tutorial

In this tutorial, we will use [RetinaNet](https://github.com/fizyr/tf-retinanet) to train head detection model on [HollywoodHeads](https://www.di.ens.fr/willow/research/headdetection/) dataset, on Azure Cloud. If you are not familiar with Azure, there is a number of services in Azure that can be used for Deep Learning:
* [Data Science Virtual Machines][DSVM], which are basically easy-to-setup Virtual Machines with everything for ML pre-installed
* [Azure Machine Learning][AzML] - a complete platform solution for ML.

Since using Azure ML requires some learning curve, we will focus this tutorial on using simple virtual machines.

### Note on the Object Detection framework

There is a number of good modern object detection model available, including Retina, YOLO, etc. In our example, we will use Keras implementation of RetinaNet. There are two version available:

* [TF RetinaNet](https://github.com/fizyr/tf-retinanet) is a newer and better modular framework, however, it is still being developed, and does not support CSV input format for the dataset.
* Older [Keras RetinaNet](https://github.com/fizyr/keras-retinanet) implementation is proven, has been used in many projects, and produce better results. We will use this version.

### HollywoodHeads Dataset

In this tutorial, we will be using [HollywoodHeads](https://www.di.ens.fr/willow/research/headdetection/) dataset, which is a dataset of frames from different Hollywood movies marked up for head detection. The dataset follows modified Pascal VOC format, with only one class - `head`. *Keras Retinanet* can use Pascal VOC format out-of-the-box, however, because there are notable differences between official Pascal VOC format and the one used in this project, it will not *just work*. In order not to modify original library, we will convert Pascal VOC format to custom CSV, also understood by *Keras Retinanet*.

### Data Prep / Training Setup

Training the model is very resource-intensive task, and GPU machine is required. However, in many real-world applications, Data Scientists spend about 80% of their time on data preparation. To minimize overall costs of the project, it make sense to use two virtual machines: GPU DSVM for training, and CPU DSVM for data preparation.

If we use two machines, we need to figure out the way to share data between them. There are several approaches that can be used:
* Using [Azure Files][AzFiles] allows you to create a SMB-mountable directory which can be shared between several machines. This sounds like an ideal solution, however, this approach results in quite slow access speed, especially in the case of many small files.
* For better access speed, you can store data in Blob storage, and then copy dataset to two VMs locally using [AzCopy][AzCopy] utility, which is very fast way to copy large amounts of data in Azure between different sources. A DSVM contains one fast temporary disk specifically for that purpose, so if your dataset is not extremely large - it might be a good way to go. However, is still results in data duplication, and if you modify or transform data in some way - you need to manually copy the changes back to storage.  
* Mounting special [Virtual Hard Disk][AzVHD] with data to both VMs. However, VM cannot be simultaneously mounted to more that one Virtual Machine, so you would only be able to access the disk from one VM at a time. It might be okay for data preparation / training scenario, but it will not work if you want to do parallel training or procrssing.
* Mounting Blob Storage directly to the VM using [Fuse Driver][Fuse]. This approach seems to be the best one, because Fuse driver is clever enough to cache files locally, yet download only those files that are actually accessed. So it will also work in the case you have several VMs, and want to organize parallel training or parallel processing. 

In our case, we will [Blob Fuse][Fuse] approach.

### Setting up the VMs and Storage

1. Set up Azure Resource Group for the project and Azure Storage Account for data:
```bash
az group create -l northeurope -n head_detect
az storage account create -l northeurope -n storage4data -g head_detect --sku Standard_LRS
```
1. Create a VM for data preparation:
```bash
az vm create 
   --resource-group head_detect --name detectvm 
   --image microsoft-dsvm:ubuntu-1804:1804:20.07.06 
   --admin-username vmuser --admin-password myp@ssw0rd2020 
   --size Standard_D3v2 
   --public-ip-address-dns-name detectvm 
   --generate-ssh-keys
```
For the simplicity, I use password access here, but you can also use SSH authentication.
1. Install blob-fuse driver:
```bash
wget https://packages.microsoft.com/config/ubuntu/18.04/packages-microsoft-prod.deb
sudo dpkg -i packages-microsoft-prod.deb
rm packages-microsoft-prod.deb
sudo apt-get update
sudo apt-get install blobfuse
```
1. Specify the storage credentials in the config text file. We will store it into `~/.fuse`:
```bash
mkdir ~/fuse
cat > ~/fuse/fuseconfig.txt
accountName storage4data
accountKey ...
containerName data
EOF
chmod 600 ~/fuse/fuseconfig.txt
```
You can get storage account key from [Azure Portal][AzPortal], or using the following:
```bash
az storage account keys list --account-name storage4data -g head_detect
```
1. We will now create a script file called `fusemount` to mount data directory using fuse driver. You can also type the commands directly, but having a script is more convenient.
1. Fuse uses some local cache directory to store files. You can create a ramdisk for faster access (more details on this [here][Fuse]), or use any fast local disk, such as SSD. In our case, we will use `/mnt/cache` as the caching directory, and `/mnt/data` as mount point. We will add commands to create those directories in the beginning of mount script. In DSVM, `/mnt` directory is not preserved, so after machine restart you would have to re-create directories and do re-mount, and having just one script to do all that is helpful:
```bash
cat > ~/fuse/fusedriver
#!/bin/bash
sudo mkdir /mnt/data /mnt/cache
sudo chown vmuser /mnt/data /mnt/cache
EOF
```
1. Now let's add the actual mounting command:
```bash
cat >> ~/fuse/fusemount
sudo blobfuse /mnt/data --tmp-path=/mnt/cache --config-file=/home/vmuser/fuse/fuseconfig.txt -o attr_timeout=240 -o entry_timeout=240 -o negative_timeout=120
EOF
chmod u+x ~/fuse/fusemount
```
1. Now you should be able to mount the storage using `~/fuse/fusemount` command.

### Downloading and Extracting Dataset

Let's download the dataset
```bash
cd /tmp
wget https://www.di.ens.fr/willow/research/headdetection/release/HollywoodHeads.zip
cd /mnt/data
jar xvf /tmp/HollywoodHeads.zip
rm /tmp/HollywoodHeads.zip
```

### Creating CSV Annotation files

It is time to start working on data preparation for training! To do this, we would need to convert existing annotations in custom Pascal VOC format into CSV format, understood by *Keras Retinanet* (described [here](https://github.com/fizyr/keras-retinanet#csv-datasets)).

To do this, we will use [mPyPl](http://shwars.github.io/mPyPl) library, which contains built-in support for handling Pascal VOC annotations. Working with Pascal VOC in mPyPl is described [here](https://github.com/shwars/mPyPl/wiki/Reading-PASCAL-VOC-Format).

In real life, the process of data preparation will take considerable amount of time. For our example, I have already prepared the script for you in this repository, so you would need first to clone it:
```bash
git clone http://github.com/CloudAdvocacy/ObjectDetection
pip -r requirements.txt
``` 
Running [`scripts/create_csv.py`](scripts/create_csv.py) would generate annotations file `annotations.txt` for you. You should also manually create `classes.csv` file:
```bash
cat > classes.csv
head,0
EOF
```

Move obtained files into `/mnt/data`:
```bash
mv *.csv /mnt/data
```

### Preparing GPU VM for training

Now we will set up second GPU-enabled virtual machine and mount data there using fuse driver.

First, let's create the VM:
We will create a VM for data preparation:
```bash
az vm create 
   --resource-group head_detect --name detectvmgpu 
   --image microsoft-dsvm:ubuntu-1804:1804:20.07.06 
   --admin-username vmuser --admin-password myp@ssw0rd2020 
   --size Standard_NC6
   --public-ip-address-dns-name detectvmgpu 
   --generate-ssh-keys
```

You would need to repeat the steps for installing **blobfuse** driver on this machine as well.

Copy `fuse` config and mount files from another machine to this one:
```bash
scp -R vmuser@detectvm.northeurope.cloudapp.azure.com:~/fuse .
```
Now mount the data directory by executing `fuse/fusemount`, and you will see the dataset available is `/mnt/data`.

### Installing Keras Retinanet

You are almost ready for training! Clone *Keras Retinanet*:
```bash
git clone https://github.com/fizyr/keras-retinanet
```
Now you should follow installation steps outlined [in Readme](https://github.com/fizyr/keras-retinanet#installation):
```bash
cd keras-retinanet
pip install numpy
pip install .
python setup.py build_ext --inplace
```

### Training

All the data should be already available under `/mnt/data`, so you can start training. It is recommended to use `screen` utility to make sure training process is not interrupted if the connection to VM is broken. To start training, use `train.py` script from within *Keras Retinanet* repository:
```bash
python keras_retinanet/bin/train.py --gpu 0 csv /mnt/data/HollywoodHeads/annotations.csv /mnt/data/HollywoodHeads/classes.csv
```

### Related Projects

* [Training Tensorflow Object Detection on Azure ML](https://github.com/liupeirong/tensorflow_objectdetection_azureml)


[DSVM]: https://azure.microsoft.com/services/virtual-machines/data-science-virtual-machines/?WT.mc_id=e2eod-github-dmitryso
[AzML]: https://azure.microsoft.com/services/machine-learning/?WT.mc_id=e2eod-github-dmitryso
[AzFiles]: https://docs.microsoft.com/azure/storage/files/storage-files-introduction/?WT.mc_id=e2eod-github-dmitryso
[AzVHD]: https://docs.microsoft.com/azure/virtual-machines/managed-disks-overview/?WT.mc_id=e2eod-github-dmitryso
[AzCopy]: https://docs.microsoft.com/azure/storage/common/storage-use-azcopy-v10/?WT.mc_id=e2eod-github-dmitryso
[Fuse]: https://docs.microsoft.com/azure/storage/blobs/storage-how-to-mount-container-linux/?WT.mc_id=e2eod-github-dmitryso
[AzPortal]: https://portal.azure.com/?WT.mc_id=e2eod-github-dmitryso