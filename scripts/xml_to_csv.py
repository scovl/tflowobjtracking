import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET

def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size/width').text),
                     int(root.find('size/height').text),
                     member[0].text,
                     int(member.find('bndbox/xmin').text),
                     int(member.find('bndbox/ymin').text),
                     int(member.find('bndbox/xmax').text),
                     int(member.find('bndbox/ymax').text))
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df

def main():
    for folder in ['train', 'test']:
        image_path = os.path.join(os.getcwd(), ('imgs/' + folder))
        xml_df = xml_to_csv(image_path)
        xml_df.to_csv(('annotations/' + folder + '_labels.csv'), index=None)
        print('Successfully converted xml to csv.')

main()
