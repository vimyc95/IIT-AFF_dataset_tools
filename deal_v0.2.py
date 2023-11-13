import cv2
import pickle
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import tqdm

type = 'test'

train_txt = 'VOCdevkit2012/VOC2012/ImageSets/Main/{}.txt'.format(type)
txt = open(train_txt, 'r')

img_dir = 'VOCdevkit2012/VOC2012/JPEGImages'
xml_dir = 'VOCdevkit2012/VOC2012/Annotations'
afford_dir = 'cache/GTsegmask_VOC_2012_train'

new_img_dir = 'VOC_affordance/JPEGImages'
new_aff_dir = 'VOC_affordance/SegmentationClass'
new_train_txt = 'VOC_affordance/ImageSets/Segmentation/{}.txt'.format(type)

new_txt = open(new_train_txt, 'w') 

for img_n in tqdm.tqdm(txt):
    img_n = img_n.strip("\n")
    img = cv2.imread("{}/{}.jpg".format(img_dir, img_n))
    tree = ET.parse('{}/{}.xml'.format(xml_dir,img_n))

    root = tree.getroot()
    afford_n = 1
    for child in root:
        if child.tag == "object":
            f = open("{}/{}_{}_segmask.sm".format(afford_dir, img_n, afford_n),'rb')
            aff = pickle.load(f)
            for obj in child:
                if obj.tag == "bndbox":
                    bbox = [int(obj[i].text) for i in range(4)]
                    # print("{}/{}".format(img_n, afford_n))

                    cv2.imwrite('{}/{}_{}.jpg'.format(new_img_dir, img_n, afford_n), img[bbox[1]:bbox[3], bbox[0]:bbox[2],:])
                    
                    aff_3 = cv2.merge((aff[bbox[1]:bbox[3], bbox[0]:bbox[2]],aff[bbox[1]:bbox[3], bbox[0]:bbox[2]],aff[bbox[1]:bbox[3], bbox[0]:bbox[2]]))
                    cv2.imwrite('{}/{}_{}.png'.format(new_aff_dir, img_n, afford_n), aff_3)
                    print(aff_3.shape)

                    new_txt.write('{}_{}\n'.format(img_n, afford_n))
                    # cv2.imshow("aff", aff[bbox[1]:bbox[3], bbox[0]:bbox[2]]*50)
                    # cv2.imshow("img", img[bbox[1]:bbox[3], bbox[0]:bbox[2],:])
                    # cv2.waitKey(0)

                    # print("{}/{}".format(img_n, afford_n))
            afford_n += 1

new_txt.close()
