#check https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md
conda activate Trident

DataPath=[your_data_path]
ProjectPath=[your_project_path]
CityScapesPath= $DataPath/CityScapes
COCOStuffPath=$DataPath/COCOStuff
COCOObjectPath=$DataPath/COCOObject
ContextPath=$DataPath/VOCdevkit

# CityScapes
pip install cityscapesscripts
python datasets/city_scapes.py $CityScapesPath --nproc 8

# COCOStuff and COCOObject
python datasets/coco_stuff164k.py $COCOStuffPath --nproc 8
python datasets/cvt_coco_object.py $COCOStuffPath -o $COCOObjectPath

# Context and Context59
cd $ContextPath/VOC2010
wget https://codalabuser.blob.core.windows.net/public/trainval_merged.json
cd $ProjectPath
cd ..
git clone https://github.com/zhanghang1989/detail-api.git
pip install Cython
cd detail-api/PythonAPI
make install
cd $ProjectPath
python datasets/pascal_context.py $ContextPath $ContextPath/VOC2010/trainval_merged.json

#ADE20K, VOC20 and VOC21 no need to process
