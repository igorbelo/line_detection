import xml.etree.ElementTree as ET

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
