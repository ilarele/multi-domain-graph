all_classes_occurences_ade = [
    (0.0, 549870074), (5.0, 196085021), (3.0, 130539277), (14.0, 57923488),
    (8.0, 41506029), (10.0, 38603593), (23.0, 18925060), (38.0, 15416607),
    (15.0, 15124148), (7.0, 12234036), (22.0, 11429933), (19.0, 10943232),
    (1.0, 6290361), (30.0, 6262136), (24.0, 5909215), (18.0, 5713509),
    (59.0, 5369725), (28.0, 4721785), (4.0, 3909738), (27.0, 3280680),
    (42.0, 2701446), (39.0, 2670298), (49.0, 2570622), (45.0, 2492901),
    (37.0, 2218113), (2.0, 2081942), (35.0, 1996099), (36.0, 1960913),
    (50.0, 1951255), (57.0, 1630938), (58.0, 1622800), (53.0, 1536057),
    (47.0, 1391282), (64.0, 1332222), (44.0, 1283270), (63.0, 1247191),
    (73.0, 1212990), (70.0, 1202752), (17.0, 1186359), (71.0, 1001847),
    (85.0, 874339), (31.0, 802744), (69.0, 753764), (121.0, 709733),
    (41.0, 667199), (11.0, 662240), (13.0, 634054), (82.0, 617574),
    (135.0, 570853), (33.0, 568563), (81.0, 535881), (118.0, 534498),
    (139.0, 484158), (124.0, 483574), (97.0, 458518), (89.0, 452971),
    (55.0, 409462), (110.0, 395923), (133.0, 317379), (92.0, 304844),
    (65.0, 292728), (117.0, 270281), (95.0, 269361), (146.0, 264597),
    (6.0, 242926), (66.0, 239119), (100.0, 224468), (67.0, 223510),
    (21.0, 188272), (40.0, 182143), (142.0, 170561), (77.0, 164337),
    (62.0, 143480), (107.0, 136812), (9.0, 130082), (148.0, 128082),
    (112.0, 127847), (141.0, 118195), (129.0, 117848), (147.0, 117666),
    (132.0, 104225), (125.0, 101879), (32.0, 101275), (137.0, 100382),
    (131.0, 83629),
    (25.0, 82969), (75.0, 66511), (134.0, 65855), (43.0, 61356), (74.0, 54641),
    (143.0, 49683), (94.0, 48872), (138.0, 48028),
    (99.0, 46676), (144.0, 46411), (98.0, 31416), (26.0, 29338), (16.0, 29163),
    (145.0, 27637), (108.0, 23065), (130.0, 22452), (93.0, 21959),
    (46.0, 19635), (127.0, 16455), (34.0, 15702), (56.0, 14740), (54.0, 13433),
    (115.0, 12659), (61.0, 11729), (120.0, 11551), (20.0, 9595), (122.0, 8760),
    (48.0, 8723), (12.0, 8223), (76.0, 7566), (52.0, 5115), (90.0, 4067),
    (84.0, 3440), (87.0, 2077), (104.0, 2034), (72.0, 1985), (51.0, 1668),
    (86.0, 1128), (105.0, 942), (106.0, 841), (111.0, 507), (116.0, 420),
    (102.0, 382), (119.0, 360), (88.0, 300), (114.0, 262), (83.0, 260),
    (123.0, 153), (101.0, 132), (126.0, 124), (149.0, 78), (78.0, 46),
    (60.0, 35), (29.0, 33), (140.0, 30), (96.0, 15), (103.0, 3)
]

LABELS_NAME = [
    'wall', 'building;edifice', 'sky', 'floor;flooring', 'tree', 'ceiling',
    'road;route', 'bed', 'windowpane;window', 'grass', 'cabinet',
    'sidewalk;pavement', 'person;individual;someone;somebody;mortal;soul',
    'earth;ground', 'door;double;door', 'table', 'mountain;mount',
    'plant;flora;plant;life', 'curtain;drape;drapery;mantle;pall', 'chair',
    'car;auto;automobile;machine;motorcar', 'water', 'painting;picture',
    'sofa;couch;lounge', 'shelf', 'house', 'sea', 'mirror',
    'rug;carpet;carpeting', 'field', 'armchair', 'seat', 'fence;fencing',
    'desk', 'rock;stone', 'wardrobe;closet;press', 'lamp',
    'bathtub;bathing;tub;bath;tub', 'railing;rail', 'cushion',
    'base;pedestal;stand', 'box', 'column;pillar', 'signboard;sign',
    'chest;of;drawers;chest;bureau;dresser', 'counter', 'sand', 'sink',
    'skyscraper', 'fireplace;hearth;open;fireplace', 'refrigerator;icebox',
    'grandstand;covered;stand', 'path', 'stairs;steps', 'runway',
    'case;display;case;showcase;vitrine',
    'pool;table;billiard;table;snooker;table', 'pillow', 'screen;door;screen',
    'stairway;staircase', 'river', 'bridge;span', 'bookcase', 'blind;screen',
    'coffee;table;cocktail;table',
    'toilet;can;commode;crapper;pot;potty;stool;throne', 'flower', 'book',
    'hill', 'bench', 'countertop',
    'stove;kitchen;stove;range;kitchen;range;cooking;stove', 'palm;palm;tree',
    'kitchen;island',
    'computer;computing;machine;computing;device;data;processor;electronic;computer;information;processing;system',
    'swivel;chair', 'boat', 'bar', 'arcade;machine',
    'hovel;hut;hutch;shack;shanty',
    'bus;autobus;coach;charabanc;double-decker;jitney;motorbus;motorcoach;omnibus;passenger;vehicle',
    'towel', 'light;light;source', 'truck;motortruck', 'tower',
    'chandelier;pendant;pendent', 'awning;sunshade;sunblind',
    'streetlight;street;lamp', 'booth;cubicle;stall;kiosk',
    'television;television;receiver;television;set;tv;tv;set;idiot;box;boob;tube;telly;goggle;box',
    'airplane;aeroplane;plane', 'dirt;track',
    'apparel;wearing;apparel;dress;clothes', 'pole', 'land;ground;soil',
    'bannister;banister;balustrade;balusters;handrail',
    'escalator;moving;staircase;moving;stairway',
    'ottoman;pouf;pouffe;puff;hassock', 'bottle', 'buffet;counter;sideboard',
    'poster;posting;placard;notice;bill;card', 'stage', 'van', 'ship',
    'fountain', 'conveyer;belt;conveyor;belt;conveyer;conveyor;transporter',
    'canopy', 'washer;automatic;washer;washing;machine', 'plaything;toy',
    'swimming;pool;swimming;bath;natatorium', 'stool', 'barrel;cask',
    'basket;handbasket', 'waterfall;falls', 'tent;collapsible;shelter', 'bag',
    'minibike;motorbike', 'cradle', 'oven', 'ball', 'food;solid;food',
    'step;stair', 'tank;storage;tank', 'trade;name;brand;name;brand;marque',
    'microwave;microwave;oven', 'pot;flowerpot',
    'animal;animate;being;beast;brute;creature;fauna',
    'bicycle;bike;wheel;cycle', 'lake',
    'dishwasher;dish;washer;dishwashing;machine',
    'screen;silver;screen;projection;screen', 'blanket;cover', 'sculpture',
    'hood;exhaust;hood', 'sconce', 'vase',
    'traffic;light;traffic;signal;stoplight', 'tray',
    'ashcan;trash;can;garbage;can;wastebin;ash;bin;ash-bin;ashbin;dustbin;trash;barrel;trash;bin',
    'fan', 'pier;wharf;wharfage;dock', 'crt;screen', 'plate',
    'monitor;monitoring;device', 'bulletin;board;notice;board', 'shower',
    'radiator', 'glass;drinking;glass', 'clock', 'flag'
]

combine_list = [(19, 30, 75, 31), (14, 58), (15, 64), (38, 96, 59, 53),
                (10, 35, 44)]
to_keep_list = [0, 5, 3, 14, 8, 10, 23, 38, 15, 7, 22, 19]
# self.to_keep_list = [0, 5, 3, 14, 8, 10, 23, 38, 15, 7, 22, 19]
#  0 [                          wall] occ 46.14% (din 100%: 46.14%)
#   5 [                       ceiling] occ 16.45% (din 100%: 62.59%)
#   3 [                floor;flooring] occ 10.95% (din 100%: 73.55%)
#  14 [door;double;door+screen;door;screen] occ 5.00% (din 100%: 78.54%)
#   8 [             windowpane;window] occ 3.48% (din 100%: 82.02%)
#  10 [cabinet+wardrobe;closet;press+chest;of;drawers;chest;bureau;dresser] occ 3.51% (din 100%: 85.54%)
#  23 [             sofa;couch;lounge] occ 1.59% (din 100%: 87.13%)
#  38 [railing;rail+escalator;moving;staircase;moving;stairway+stairway;staircase+stairs;steps] occ 1.87% (din 100%: 89.00%)
#  15 [table+coffee;table;cocktail;table] occ 1.38% (din 100%: 90.38%)
#   7 [                           bed] occ 1.03% (din 100%: 91.41%)
#  22 [              painting;picture] occ 0.96% (din 100%: 92.37%)
#  19 [chair+armchair+swivel;chair+seat] occ 1.52% (din 100%: 93.88%)

import pdb
pdb.set_trace()