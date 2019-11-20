import os
import xml.etree.ElementTree as ET
import fnmatch
import numpy as np
from pprint import pprint
import pdb


from data.util import read_image


class ImgVidDataset:
    def __init__(self, data_dir, split='train', num=1):
        self.tw = 5
        self.all_ids = []

        # because some of images only have one bounding box which is occluded
        self.use_occluded = True

        all_num = 0
        for ii in range(num):
            ii += 1
            print('ii: ', ii)
            
            id_list_file = os.path.join(data_dir, 'ImageSets/VID/' + split + '_' + str(ii) + '.txt')
            print('id_list_file: ', id_list_file)


            cur_ids = [id_.strip()[:-2] for id_ in open(id_list_file)]
            print('cur_ids: ', cur_ids)

            cur_lbl = ii
            print('lbl: ', cur_lbl)

            for name in cur_ids:
                dirpath = './datasets/ILSVRC2015/Data/VID/' + split + '/' + name
                num_imgs = len(fnmatch.filter(os.listdir(dirpath), '*.JPEG'))
                print('name: ', name, ' num_images: ', num_imgs)

                num_usable = int(num_imgs/self.tw) * self.tw
                print('num_usuable: ', num_usable)

                all_num += int(num_imgs/self.tw)

                #continue

                for jj in range(num_usable):
                    #print(jj)
                    if jj % self.tw == 0:
                        if jj != 0:
                            #print('append => ', jj-1)
                            self.all_ids.append(cur_id)

                        cur_id = []
                        cur_id.append(dirpath + '/' + format(jj, '06'))
                    else:
                        cur_id.append(dirpath + '/' + format(jj, '06'))
                self.all_ids.append(cur_id) # last set

                print('len all ids: ', len(self.all_ids))
                #pprint(all_ids)

                print('all_num: ', all_num)
                #exit(0)


    def __len__(self):
        return len(self.all_ids)


    def get_example(self, i):

        id_ = self.all_ids[i]

        imgs = []
        bboxes = []
        labels = []
        occludeds = []

        for name in id_:
            print('\n TTTTTT \n')
            file_name = name + '.JPEG'
            print('IMG PATH: ', file_name)

            img = read_image(file_name, color=True)
            #imgs.append(img)

            anno_path = name.replace('Data', 'Annotations') + '.xml'
            print('ANNOT PATH: ', anno_path)

            anno = ET.parse(anno_path)

            bbox = list()
            label = list()
            occluded = list()
            
            for obj in anno.findall('object'):
                # when in not using difficult split, and the object is
                # difficult, skipt it.
                if not self.use_occluded and int(obj.find('occluded').text) == 1:
                    continue

                occluded.append(int(obj.find('occluded').text))
                bndbox_anno = obj.find('bndbox')
                # subtract 1 to make pixel indexes 0-based
                # bbox.append([
                #     int(bndbox_anno.find(tag).text) - 1
                #     for tag in ('ymin', 'xmin', 'ymax', 'xmax')])
                tmp_bbox = [int(bndbox_anno.find(tag).text) - 1
                    for tag in ('xmin', 'ymin', 'xmax', 'ymax')]
                name = obj.find('name').text.lower().strip()
                label.append(VOC_BBOX_LABEL_NAMES.index(name))
                tmp_bbox.append(int(VOC_BBOX_LABEL_NAMES.index(name)))
                bbox.append(tmp_bbox)

            imgs.append(img)
            bboxes.append(bbox)
            labels.append(label)
            occludeds.append(occluded)

        
        imgg = np.stack(imgs).astype(np.uint8)
        boxes = np.stack(bboxes).astype(np.float32)
        lbls = np.stack(labels).astype(np.uint8)
        occls = np.array(occludeds, dtype=np.bool).astype(np.uint8)

        print('\n ET EXAMPLE IN FIRST STEP\n\n')
        print(imgg.shape)
        print(boxes.shape)
        print(lbls.shape)
        print(occls.shape)
        print('\nQQQQQQQQQQQQQQQQQQQQQQQQQ\n')


        return imgg, boxes, lbls, occls

                    
VOC_BBOX_LABEL_NAMES_REAL = (
    'airplane',
    'antelope',
    'bear',
    'bicycle',
    'bird',
    'bus',
    'car',
    'cattle',
    'dog',
    'domestic_cat',
    'elephant',
    'fox',
    'giant_panda',
    'hamster',
    'horse',
    'lion',
    'lizard',
    'monkey',
    'motorcycle',
    'rabbit',
    'red_panda',
    'sheep',
    'snake',
    'squirrel',
    'tiger',
    'train',
    'turtle',
    'watercraft',
    'whale',
    'zebra'
)


VOC_BBOX_LABEL_NAMES = (
    'n02691156',
    'n02419796',
    'n02131653',
    'n02834778',
    'n01503061',
    'n02924116',
    'n02958343',
    'n02402425',
    'n02084071',
    'n02121808',
    'n02503517',
    'n02118333',
    'n02510455',
    'n02342885',
    'n02374451',
    'n02129165',
    'n01674464',
    'n02484322',
    'n03790512',
    'n02324045',
    'n02509815',
    'n02411705',
    'n01726692',
    'n02355227',
    'n02129604',
    'n04468005',
    'n01662784',
    'n04530566',
    'n02062744',
    'n02391049'
)


#if __name__ == '__main__':
#    test = ImgVidDataset('./ILSVRC2015_VID_initial/ILSVRC2015/')
#    ff = test.get_example(0)
#    pdb.set_trace()


