from modelscope.msdatasets import MsDataset
ds = MsDataset.load("coco_2014_caption", namespace="modelscope", split="train")
print(ds[0])