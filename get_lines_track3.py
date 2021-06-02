import os

# Retrieves the video id
def get_video_id(path):
    video_id_dict = {}
    print()
    with open(os.path.join(path,'list_video_id.txt'),'r') as f:
        for line in f:
            line = line.rstrip()
            video = line.split(' ')
            video_id_dict[int(video[0])] = video[1]
    return video_id_dict

# Retrueves hard cided Region of Interest from Track 3
def get_rois(cam_id,data_path):
    print("cam_id:", cam_id)
    #cam_path = os.path.join(data_path, 'ROIs/cam_{}.txt'.format(cam_id))
    cam_path = os.path.join(data_path, 'ROIs/{}.txt'.format(cam_id))
    with open(cam_path, 'r') as f:
        rois=[]
        for line in f:
            line = line.rstrip()
            p0 = (int(line.split(',')[0]),int(line.split(',')[1]))
            pre=p0
            break
        for line in f:
            line = line.rstrip()
            now = (int(line.split(',')[0]),int(line.split(',')[1]))
            rois.append([pre,now])
            pre=now
        rois.append([pre,p0])
    return len(rois), rois


# Retrieves hardcoded annotations about the cameras
# mov_rois is the direction and lines are when the cars intersect the line for counting
def get_lines(cam_id):
    directions = []  # valid direction for each line
    mov_rois = []  # corresponding RoI line for each movement
    print('cam_id ',cam_id)
    if cam_id == 1:
        line_2 = [(207, 738), (207, 580)]
        line_1 = [(339, 600), (339, 505)]
        line_3 = [(172, 1034), (380, 772)]
        line_4 = [(1918, 779), (1852, 779)]
        line_5 = [(1730, 391), (1730, 494)]
        line_6 = [(1514, 382), (1532, 315)]
        line_7 = [(1538, 542), (1710, 542)]
        line_8 = [(1916, 545), (1822, 546)]
        line_10 = [(992, 304), (1048, 304)]
        line_11 = [(991, 367), (869, 367)]
        line_12 = [(721, 313), (772, 379)]

        lines=[line_1,line_2,line_3,line_4, line_5, line_6, line_7, line_8, line_10, line_11, line_12]
        mov_rois = [4, 1, 1, 2]

       #t3 
    elif cam_id == 2:
        line_1 = [(1739, 754), (1916, 913)]
        line_2 = [(1901, 540), (1708, 701)]
        line_3 =   [(1767, 495), (1823, 365)]
        line_4 = [(83, 614), (28, 482)]
        line_5 = [(51, 618), (51, 963)]
        #line_6 = [] omit bc cant see cars
        line_7= [(1054, 333), (1195, 333)]
        line_8 = [(1017, 400), (820, 400)]
        line_9 = [(646, 295), (701, 413)]
        line_10 = [(1192, 1062), (523, 1062)]
        line_11 = [(1457, 1027), (1786, 1027)]
        lines = [line_1, line_2, line_3, line_4, line_5, line_7, line_8, line_9, line_10, line_11]
        directions = [2, 2, 2, 1, 1, 3, 3, 2, 4, 4] #left, left, left, right, ruight, 6, down, down, left, up, up

    #t3    
    elif cam_id == 3:
        line_12 = [(1631, 328), (1631, 208)]
        line_11 = [(1111, 449), (1111, 322)]
        line_10 = [(787, 664), (1016, 842)]
        line_9 = [(7, 758), (109, 654)]
        line_8 = [(88, 496), (88, 639)] #line 7 no exclusive edges.
        #lines 6 is too close to the edge
        line_5 = [(1641, 706), (1919, 566)]
        line_4 = [(1312, 892), (1714, 792)]
        line_3 = [(281, 400), (251, 266)]
        line_2 = [(421, 374), (800, 355)]
        line_1 = [(827, 262), (967, 262)]

        lines = [line_1, line_2, line_3, line_4, line_5, line_8, line_9, line_10, line_11, line_12]
        directions = [3, 3, 2, 4, 4, 1, 3, 3, 2, 2] #down, down, left, up, up, 6, 7, right, down, down, left, left

 #t3       
    elif cam_id == 4:
 
        line_9 = [(1582, 427), (1775, 332)]
       # line_8 = [(1103, 495),(1163, 406)] #line 8 turns from right to down, and is too close w line7 going across Right to left. 
        line_7 = [(1897, 467), (1897, 568)] #no exclusive area betwen 8 and 7
        line_6 = [(142, 453), (266, 541)]
        line_5 = [(797, 530), (378, 530)]
        line_4 = [(1000, 374), (847, 379)]
       # line_3 = [(), ()] omitted no cars drive this path
        line_2 = [(1087, 830), (1913, 752)]
        line_1 = [(1234, 1010), (1634, 888)]
        directions = [4, 4, 3, 3, 2, 2, 2]#up, up, 3, down, down, left, left, 8, left
        lines = [line_1, line_2, line_4, line_5, line_6, line_7, line_9]
    
    #t3
    elif cam_id == 5:
        line_1 = [(1211, 294), (1278, 328)]
        line_2 = [(944, 247), (1001, 313)]
        line_3 = [(896, 170), (868, 241)]
        line_4 = [(512, 876), (662, 709)]
        line_5 = [(375, 662), (398, 503)]
        line_6 = [(435, 491), (401, 415)]
        line_7 = [(565, 232), (624, 223)]
        line_8 = [(509, 268), (552, 294)]
        line_9 = [(396, 256), (476, 284)]
        line_10 = [(1380, 886), (14892, 804)]
        line_11 = [(1584, 708), (1503, 790)]
        line_12 = [(1457, 497), (1607, 480)]
        directions = [2, 2, 2, 2, 3, 1, 1, 3, 3, 2, 4, 4, 4] #left, left , left ,left, down,right, right, down, down, left, up, up, up
        lines = [line_1, line_2, line_3, line_4, line_5, line_6, line_7, line_8, line_9]
        
    elif cam_id == 10: # Cam 10
        line_1 = [(575, 376), (808, 355)]
        line_2 = [(297, 520), (520, 223)]
        #directions = [3, 3, 3]
        mov_rois = [4, 3] #unchanged
        lines =  [line_1, line_2]

#t3
    elif cam_id == 11: # Cam 11
        line_1 = [(220, 429), (446, 608)]
        line_2 = [(395, 335), (506, 332)]
        directions = [1, 2]
        mov_rois = [3, 2]
        lines = [line_1, line_2]
 
 #t3
    elif cam_id == 12: # Cam 12
        line_1 = [(117, 624), (247, 415)]
        line_2 = [(500, 435), (500, 303)]
        directions = [1, 2]
        mov_rois = [4, 3]
        lines = [line_1, line_2]

#t3
    elif cam_id == 13: # Cam 13
        line_1 = [(335, 373), (509, 303)]
        line_2 = [(630, 329), (720, 245)]
        directions = [1, 2]
        mov_rois = [3, 2]
        lines = [line_1, line_2]

    # elif cam_id == 14:
    #     line_1 = [(242, 1018), (531, 1377)]
    #     line_2 = [(1598, 1312), (2079, 1531)]
    #     mov_rois = [3, 1]
    #     lines = [line_1, line_2]

    # elif cam_id == 15:
    #     line_1 = [(965, 575), (1286, 433)]
    #     line_2 = [(845, 312), (1045, 248)]
    #     mov_rois = [2, 0]
    #     lines = [line_1, line_2]

    # elif cam_id == 16:
    #     line_1 = [(236, 657), (947, 665)]
    #     line_2 = [(824, 211), (1171, 216)]
    #     mov_rois = [2, 0]
    #     lines = [line_1, line_2]

    # elif cam_id == 17:
    #     line_1 = [(796, 843), (1519, 721)]
    #     line_2 = [(812, 227), (1209, 216)]
    #     mov_rois = [2, 0]
    #     lines = [line_1, line_2]

    # elif cam_id == 18:
    #     line_1 = [(225, 832), (744, 856)]
    #     line_2 = [(923, 285), (1175, 289)]
    #     mov_rois = [2, 0]
    #     lines = [line_1, line_2]

    # elif cam_id == 19:
    #     line_1 = [(585, 786), (1039, 800)]
    #     line_2 = [(1048, 647), (1491, 644)]
    #     mov_rois = [2, 0]
    #     lines = [line_1, line_2]

    # elif cam_id == 20:
    #     line_1 = [(220, 586), (855, 672)]
    #     line_2 = [(1018, 655), (1525, 694)]
    #     mov_rois = [2, 0]
    #     lines = [line_1, line_2]

#t3
    elif cam_id == 18: # Cam 18
        line_1 = [(719, 1455), (1243, 1741)]
        line_2 = [(1420, 1264), (1329, 1133)]
        directions = [1, 2] #right, left
        lines = [line_1, line_2]

    elif cam_id == 19:
        line_1 = [(551, 450), (551, 700)]
        line_2 = [(788, 750), (788, 1031)]        
        directions = [1,2] #right, left
        lines = [line_1, line_2]

    elif cam_id == 20:
        line_1 = [(723, 664), (971, 542)]
        line_2 = [(1114, 573), (1251, 463)]        
        directions = [1,2] #right, left
        lines = [line_1, line_2]

    elif cam_id == 22:
        line_1 = [(1246, 658), (876, 686)]
        line_2 = [(802, 640), (441, 640)]        
        directions = [4,3] #up, down
        lines = [line_1, line_2]

    elif cam_id == 23:
        line_1 = [(912, 842), (709, 686)]
        line_2 = [(919, 605), (811, 510)]        
        directions = [1,2] #right, left
        lines = [line_1, line_2]


#t3
    elif cam_id == 33: # Cam 33
         line_1 = [(614,351), (1040, 351)]
         line_2 = [(189, 330), (580, 330)]
         lines = [line_1, line_2] 
#t3
    elif cam_id == 34: # Cam 34
        line_1 = [(577, 445), (645, 190)]
        line_2 = [(543, 240), (434, 314)]
        directions = [2, 1]  #left, right
        lines = [line_1, line_2]
#t3
    elif cam_id == 35: # Cam 35
        line_1 = [(806, 83), (870, 199)]
        line_2 = [(1170, 166), (1076, 137)]
        line_3 = [(1380, 350), (1596, 311)]
        line_4 = [(1235, 587), (1316, 397)]
        line_5 = [(637, 859), (839, 524)]
        line_6 = [(641, 285), (833, 438)]
        mov_rois = []
        directions = [2, 3, 4, 2, 1, 1] #left, down, up, left, right, right
        lines = [line_1, line_2, line_3, line_4, line_5, line_6]

#t3
    elif cam_id == 38: # Cam 38
        line_1 = [(1151, 735), (1350, 933)]
        line_2 = [(1592, 964), (2172, 1158)]
        mov_rois = []
        directions = [2, 1] # left, right

#t3
    elif cam_id == 39: # Cam 39
        line_1 = [(590, 581), (442, 399)]
        line_2 = [(1031, 1069), (1031, 585)]
        line_3 = [(393, 268), (260, 285)]
        mov_rois = []
        directions = [2, 1, 4] # left, right, up
#t3
    elif cam_id == 40: # Cam 40
        line_1 = [(1029, 445), (874, 556)]
        line_2 = [(475, 702), (761, 578)]
        line_3 = [(1471, 347), (1668, 324)]
        mov_rois = [] 
        directions = [2, 1, 4] # left, right, up 
    print(lines)
    return len(lines), lines, directions, mov_rois


