import os
import cv2


att_dict = {'5_o_Clock_Shadow': 0, 'Arched_Eyebrows': 1, 'Attractive': 2,
            'Bags_Under_Eyes': 3, 'Bald': 4, 'Bangs': 5, 'Big_Lips': 6,
            'Big_Nose': 7, 'Black_Hair': 8, 'Blond_Hair': 9, 'Blurry': 10,
            'Brown_Hair': 11, 'Bushy_Eyebrows': 12, 'Chubby': 13,
            'Double_Chin': 14, 'Eyeglasses': 15, 'Goatee': 16,
            'Gray_Hair': 17, 'Heavy_Makeup': 18, 'High_Cheekbones': 19,
            'Male': 20, 'Mouth_Slightly_Open': 21, 'Mustache': 22,
            'Narrow_Eyes': 23, 'No_Beard': 24, 'Oval_Face': 25,
            'Pale_Skin': 26, 'Pointy_Nose': 27, 'Receding_Hairline': 28,
            'Rosy_Cheeks': 29, 'Sideburns': 30, 'Smiling': 31,
            'Straight_Hair': 32, 'Wavy_Hair': 33, 'Wearing_Earrings': 34,
            'Wearing_Hat': 35, 'Wearing_Lipstick': 36,
            'Wearing_Necklace': 37, 'Wearing_Necktie': 38, 'Young': 39,
            'HairLength': 40}


test_image_dir = '/dockerdata/home/rpf/rpf/yuncao/dst/'
test_image_att_list = '/dockerdata/home/rpf/rpf/xmmtyding/test_data/dst.att_list.txt'
resiezed_dir = '/dockerdata/home/rpf/rpf/xmmtyding/test_data/dst_rs384'


destroyed_ = []
image_list = [x for x in os.listdir(test_image_dir) if x.endswith('_res.png')]
if not os.path.exists(resiezed_dir):
    os.mkdir(resiezed_dir)
for i, image_name in enumerate(image_list):
    src = os.path.join(test_image_dir, image_name)
    dst = os.path.join(resiezed_dir, image_name)
    try:
        img = cv2.imread(src)
        img = cv2.resize(img, (384, 384))
    except:
        destroyed_.append(src)

    cv2.imwrite(dst, img)
    print("{}/{}".format(i, len(image_list)), end='\r')

print(len(destroyed_))
print(destroyed_)

destroy_set = set(destroyed_)
image_list = [x for x in os.listdir(test_image_dir) if x.endswith(
    '_res.png') and x not in destroy_set]

fo = open(test_image_att_list, 'w')
fo.write('{}\n'.format(len(image_list)))
cols = ' '.join(list(att_dict.keys()))
fo.write(cols + '\n')

atts_label = ['-1'] * len(att_dict.keys())
atts_line = ' '.join(atts_label)

for image_name in image_list:
    line = image_name + ' ' + atts_line + '\n'
    fo.write(line)
fo.close()
