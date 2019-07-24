import numpy as np
import xml.etree.ElementTree as ET

filepath_xml = "C:/Users/sjn/Documents/QZ summer/testing_xml_parsing.xml"
filepath_s = "C:/Users/sjn/Documents/GitHub/qz_summer_2019/Calibration QTM2OR/data_files/s_from_calibration.txt"
s = np.loadtxt(filepath_s) # [m]
s *= 1000 # [mm]
tree = ET.parse(filepath_xml)
root = tree.getroot()

for body in root:
	name = body.find('Name')
	if name.text == 'oculus_rift':
		points = body.find('Points')
		for point in points:
			point.attrib['X'] = str(float(point.attrib['X']) + s[0])
			point.attrib['Y'] = str(float(point.attrib['Y']) + s[1])
			point.attrib['Z'] = str(float(point.attrib['Z']) + s[2])

tree.write(filepath_xml)
