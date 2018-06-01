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

    model = FcnModel('/content/fcn/output/train')
    mious = 308.32368341088295
    for i in range(583, 1449):
        miou = model.eval(datas[i], labels[i])
        mious += miou
        print('{0}/{1}, miou: {2}'.format(i, len(labels), miou)) 
        print(mious)
  
    print(mious / len(labels))


    


if __name__ == '__main__':
    main()