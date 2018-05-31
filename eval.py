from prediction import FcnModel
import os



def main():
    root = './VOCdevkit/VOC2012' 
    txt_fname = os.path.join(root, 'ImageSets/Segmentation/val.txt')
    orig_dir = os.path.join(root, 'JPEGImages')
    with open(txt_fname, 'r') as f:
        images = f.read().split() 
    datas = []
    labels = []
    for img in images:
        datas.append('%s/JPEGImages/%s.jpg' % (root, img))
        labels.append('%s/SegmentationClass/%s.png' % (root, img))
    pairs = zip(datas, labels)
    
    model = FcnModel('/content/fcn/output/train')
    count = 0
    mious = 0
    for (img, lable) in pairs:
        miou = model.eval(img, lable)
        mious += miou
        count += 1
        print('{0}/{1}, miou: {2}'.format(count, len(labels), miou)) 
        print(mious)
    print(mious / len(labels))


    


if __name__ == '__main__':
    main()