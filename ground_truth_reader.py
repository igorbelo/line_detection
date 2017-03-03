import xml.etree.ElementTree as ET
import math

def build_line_meta(file):
    tree = ET.parse(file)
    root = tree.getroot()
    DL_PAGE = root[0][0]

    line_meta = {}
    last_line = None
    for zone in DL_PAGE:
        lineID = zone.attrib['lineID']
        row = int(zone.attrib['row'])
        endRow = int(zone.attrib['row']) + int(zone.attrib['height'])
        if lineID != last_line:
            line_meta[lineID] = {
                'startRow': row,
                'endRow': endRow
            }
        else:
            if line_meta[lineID]['startRow'] > row:
                line_meta[lineID]['startRow'] = row

            if line_meta[lineID]['endRow'] < endRow:
                line_meta[lineID]['endRow'] = endRow
        last_line = zone.attrib['lineID']

    return line_meta

def build_new_resolution(file):
    tree = ET.parse(file)
    root = tree.getroot()
    DL_PAGE = root[0][0]

    line_meta = {}
    last_line = None
    for zone in DL_PAGE:
        lineID = zone.attrib['lineID']
        row = int(zone.attrib['row'])
        col = int(zone.attrib['col'])
        width = int(zone.attrib['width'])
        height = int(zone.attrib['height'])

        zone.set('row', str(int(math.ceil(row * 0.09803921568627))))
        zone.set('col', str(int(math.ceil(col * 0.09803921568627))))
        zone.set('width', str(int(math.ceil(width * 0.09803921568627))))
        zone.set('height', str(int(math.ceil(height * 0.09803921568627))))

    tree.write(file)

if __name__ == '__main__':
    build_new_resolution(
        "ground-truth/LOW_AAW_ARB_20061101.0057-S1_2_LDC0229.xml"
    )
