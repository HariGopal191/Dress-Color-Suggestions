import os,sys
if sys.version_info[0] == 2:
    print
else:
    print("This Application supports only python --version less than '2.7' only")
    sys.exit(-1)
import ConfigParser as cp
import platform
import argparse as ap
import os.path
import mysql.connector
import numpy 
import cv2
from PIL import Image, ImageDraw
import PIL
from imutils import face_utils
import argparse
import imutils
import dlib
import random
import fnmatch , os ,colorsys

def file_exists(fname):
        if os.path.isfile(fname):
                return
        else:
                print('file', fname, ' doesnt exist, it is a mandatory file - please check your setup, please check the config file you are using and update the respective field there,  exiting...')
                sys.exit(-1)

def conf_initiator(conf_file):
        config = cp.ConfigParser()
        config.read(conf_file)          #       change this to invoke inner modules
        #config._sections
        developer = dict(config._sections['mysqldb'])
        return  developer

def parse_initiator():
        parser = ap.ArgumentParser()
        parser.add_argument('-c', '--conf',action='store', dest='conf_file', help='Specify --conf <path of the configuration file> ', required=False)
        parser.add_argument('-b', '--bulk_insertion',action='store', dest='bulk_insertion', help='Specify --bulk_insertion <specify directory of the images folder> ', required=False)
        parser.add_argument('-i', '--single_image',action='store', dest='single_image', help='Specify --single_image <specify the path of a single image>', required=False)

        results = parser.parse_args()
        try:
                if results.conf_file:
                    conf_file=results.conf_file
                else:
                    conf_file="/home/harigopal/git/pvpsit/conf/mysql.conf"
                    file_exists(conf_file)
                #print(conf_file)
                #global conf_file

        except:
                print('no config file specified')
                parser.print_help()
                sys.exit(-1)

        return results,conf_initiator(conf_file)


#class_1
class raw_image:

    #__init__
    def __init__(self, dir_name):
        self.dir_name=dir_name


    def extract_properties_single_image (self,img,config_dict,insertion):
        my_db                   =   self.connect_to_pvpsit(config_dict)
        file_name           =   img.split('.')[0]
        #print('Hello',img)
        if insertion==2:
            file_name = os.path.splitext(self.dir_name)
            file_name = file_name[0].split('/')[-1]
            print(file_name)
        img                 =   self.dir_name+img
        output_path             =   config_dict['output_path']
        #print output_path
        frame_path = config_dict['frame']
        process_path            =   config_dict['process_path']
        face_casc               =   config_dict['face_casc']
        #print face_casc

        image_properties    =   self.extract_image_properties(img)
        print(image_properties)
        faces            =   self.face_shape(img,file_name,process_path)
        #print(faces)
        features            =   self.color_skin_hair(faces[0],faces[1])
        #obtain (r,g,b) values of upper_lower_body dress
        upper_color,lower_color,ubody_name,lwbody_name =   self.upper_lower_color(img,config_dict,file_name)
        #covert them into hsv
        upper_hsv          =   colorsys.rgb_to_hsv(upper_color[0]/255.,upper_color[1]/255.,upper_color[2]/255.) 
        lower_hsv          =   colorsys.rgb_to_hsv(lower_color[0]/255.,lower_color[1]/255.,lower_color[2]/255.)
        #print(upper_color,lower_color)
        up_hsv=int(upper_hsv[0]*360)
        lw_hsv=int(lower_hsv[0]*360)
        features           =   features+(up_hsv,lw_hsv)
        print(features)
        suggestions         =   ((up_hsv,lw_hsv),)+self.return_suggestions(features,my_db)
        #print(suggestions)
        reg_name,inv_name,face_name       =   self.Shift_hue(img,file_name,config_dict,faces[2])
        #blend               =   self.Shift_hue(lwbody_name,file_name,config_dict,0)
        suggestions1 = []
        for x in suggestions:
            if x[0]>30:
                suggestions1.append(x)
        print(suggestions1)
        for i in range(0,len(suggestions1)):
            #print(i)
            #amount = suggestions[i][0]-suggestions[0][0]
            img2                =   self.hueShift(inv_name, suggestions1[i][0]/360.)
            img2.save(process_path+file_name+'_hue.jpg')
            img2 = cv2.imread(process_path+file_name+'_hue.jpg')
            img1 = cv2.imread(reg_name)
            img2 =img2 +img1
            img3 = cv2.imread(face_name)
            (c,r)= img2.shape[0:2]

            img2[0:faces[2] , 0:r] = img3

            if insertion ==1:
                cv2.imwrite('{}{}_hue{}.jpg'.format(output_path,file_name,i),img2)
            elif insertion ==2:
                cv2.imwrite(frame_path+'frame_picture{}.png'.format(i),img2)
            elif insertion==3:
                cv2.imwrite(frame_path+'frame_frame{}.png'.format(i),img2)
        if insertion ==1:
            return image_properties,features,suggestions1
        elif insertion ==2:
            #print(image_properties)
            #print(features)
            #print(suggestions)
            sys.exit(-1)
        elif insertion ==3:
            return True

    #extract_properties_bulk_images
    def extract_properties_bulk_images (self,config_dict,insertion):
        list_of_files           =   self.listOfImages()
        my_db                   =   self.connect_to_pvpsit(config_dict)
        #print(list_of_files)
        for img in list_of_files:
            image_properties,features,suggestions = self.extract_properties_single_image(img,config_dict,insertion)
            insert              =   self.insert_into_database(my_db,image_properties,features,suggestions)
        sys.exit(-1)
           
    

    def insert_into_database(self,my_db,image_properties,features,suggestions):
        self.insert_image_properties(image_properties, my_db)
        #print(image_properties)
        id_                 =   self.retrieve_id(image_properties[0],my_db)
        features            =   features+(id_,)
        self.insert_features(features,my_db)
        #print(features)
        fid                 =   self.retrieve_fid(id_,my_db)
        for i in range(0,len(suggestions)):
            suggestion         =   suggestions[i]+(fid,)
            #print(suggestion)
            self.insert_suggestions(suggestion,my_db)
        return True



    def capture_frame_from_camera(self,config_dict):
        cam              =   cv2.VideoCapture(0)
        while True:
            ret, frame      =   cam.read()
            #cv2.imshow("test", frame)
            cv2.imshow('openCV',frame)
            if not ret:
                break            
            k   =   cv2.waitKey(1) & 0xFF
            if k == 27:
        # ESC pressed
                print("Escape hit, closing...")
                break
            elif cv2.waitKey(30000):
        # SPACE pressed
                print("output_frame.png written!")
                cv2.imwrite(config_dict['output_path']+'opencv_frame.png',frame)
                conversion  =   img_obj.extract_properties_single_image('opencv_frame.png',config_dict,insertion)
        cam.release()
        cv2.destroyAllWindows()
        sys.exit(-1)

    def upper_lower_color(self,img,config_dict,file_name):
        image = PIL.Image.open(img)
        c,r = image.size
        image1 = image.crop((0,0,r,c/2))
        image2 = image.crop((0,c/2,r,c))
        upper_body = Image.Image.resize(image1,(150, 150),resample=0)
        lower_body = Image.Image.resize(image2,(150, 150),resample=0)
        ubody_name = config_dict['process_path']+file_name+'ubody.jpg'
        lwbody_name = config_dict['process_path']+file_name+'lwbody.jpg'
        upper_body.save(ubody_name)
        lower_body.save(lwbody_name)
        upper_color,lower_color = self.get_hue_color(img,ubody_name,lwbody_name,150)
        return (upper_color,lower_color,ubody_name,lwbody_name)


    def rgb_to_hsv(self,rgb):
        # Translated from source of colorsys.rgb_to_hsv
        # r,g,b should be a numpy arrays with values between 0 and 255
        # rgb_to_hsv returns an array of floats between 0.0 and 1.0.
        rgb = rgb.astype('float')
        hsv = numpy.zeros_like(rgb)
        # in case an RGBA array was passed, just copy the A channel
        hsv[..., 3:] = rgb[..., 3:]
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        maxc = numpy.max(rgb[..., :3], axis=-1)
        minc = numpy.min(rgb[..., :3], axis=-1)
        hsv[..., 2] = maxc
        mask = maxc != minc
        hsv[mask, 1] = (maxc - minc)[mask] / maxc[mask]
        rc = numpy.zeros_like(r)
        gc = numpy.zeros_like(g)
        bc = numpy.zeros_like(b)
        rc[mask] = (maxc - r)[mask] / (maxc - minc)[mask]
        gc[mask] = (maxc - g)[mask] / (maxc - minc)[mask]
        bc[mask] = (maxc - b)[mask] / (maxc - minc)[mask]
        hsv[..., 0] = numpy.select(
        [r == maxc, g == maxc], [bc - gc, 2.0 + rc - bc], default=4.0 + gc - rc)
        hsv[..., 0] = (hsv[..., 0] / 6.0) % 1.0
        return hsv


    def hsv_to_rgb(self,hsv):
        # Translated from source of colorsys.hsv_to_rgb
        # h,s should be a numpy arrays with values between 0.0 and 1.0
        # v should be a numpy array with values between 0.0 and 255.0
        # hsv_to_rgb returns an array of uints between 0 and 255.
        rgb = numpy.empty_like(hsv)
        rgb[..., 3:] = hsv[..., 3:]
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        i = (h * 6.0).astype('uint8')
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        i = i % 6
        conditions = [s == 0.0, i == 1, i == 2, i == 3, i == 4, i == 5]
        rgb[..., 0] = numpy.select(conditions, [v, q, p, p, t, v], default=v)
        rgb[..., 1] = numpy.select(conditions, [v, v, v, q, p, p], default=t)
        rgb[..., 2] = numpy.select(conditions, [v, p, t, v, v, q], default=p)
        return rgb.astype('uint8')

    def hueChange(self,img, hue):
        arr = numpy.array(img)
        hsv = self.rgb_to_hsv(arr)
        hsv[..., 0] = hue
        rgb = self.hsv_to_rgb(hsv)
        return Image.fromarray(rgb, 'RGB')


    def hueShift(self,img ,amount):
        img = Image.open(img).convert('RGB')
        arr = numpy.array(img)
        hsv = self.rgb_to_hsv(arr)
        hsv[..., 0] = (hsv[..., 0]+amount) % 1.0
        rgb = self.hsv_to_rgb(hsv)
        return Image.fromarray(rgb, 'RGB')

#apply the different hue_value_changes on the both upper & lower body
#then join face + upper_body + lower_body
    def Shift_hue(self,img, file_name,config_dict, face_chin):
        min_YCrCb = numpy.array([80,133,77],numpy.uint8)
        max_YCrCb = numpy.array([255,173,127],numpy.uint8)
        Image = cv2.imread(img)
        #shape=cv2.imread(config_dict['process_path']+'_shape.jpg')
        #cv2.imshow('shape',shape)
        if face_chin >0:    
            print(face_chin)
            (c , r) = Image.shape[0:2]
            #face = numpy.zeros((face_chin,r,3))
            face1 = numpy.zeros((face_chin,r,3))
            face = Image[0:face_chin , 0:r]
            face_name = config_dict['process_path']+file_name+'_face.jpg'
            cv2.imwrite(face_name,face)
            Image[0:face_chin , 0:r] = face1
        YCrCb = cv2.cvtColor(Image,cv2.COLOR_BGR2YCR_CB)
        Region = cv2.inRange(YCrCb,min_YCrCb,max_YCrCb)
        Region = cv2.GaussianBlur(Region, (3, 3), 0)
        Invert   = 255 - Region
        Invert[Invert<255] = 0
        Region_ex = cv2.bitwise_and(Image, Image, mask = Region)
        Invert_ex = cv2.bitwise_and(Image, Image, mask = Invert)
        region_name = config_dict['process_path']+file_name+'_region.jpg'
        invert_name = config_dict['process_path']+file_name+'_invert.jpg'
        cv2.imwrite(region_name,Region_ex)
        cv2.imwrite(invert_name,Invert_ex)
        
        #cv2.imshow('Reg',Region_ex)
        #cv2.imshow('Inv',Invert_ex) 
        return region_name,invert_name,face_name


    #suggestions
    def return_suggestions(self,features,mydb): 
        mycursor=mydb.cursor()
        list_suggestions = ()
        mycursor.execute("select upper_color,lower_color from features where skin_tone = %s  and hair_color = %s",('Normal','Brown',))
        #mycursor.execute("select * from features")
        myresult = mycursor.fetchall()
        #print(myresult)
        for x in myresult:
            #print(x)
            list_suggestions+=(x,)

        return list_suggestions
        
        #ls
    def listOfImages(self):
        listOfFiles = os.listdir(self.dir_name)
        pattern = "*.jpg"
        list_img=[]
        for entry in listOfFiles:
            if fnmatch.fnmatch(entry, pattern):
                list_img.append(entry)
        return list_img

    #extract image properties
    def extract_image_properties(self,img):
        #cwd    =  os.getcwd()
        #path   =  cwd + '/' + self.infile
        size   =  os.path.getsize(img)
        extension  =  os.path.splitext(img)[1][1:]
        record    =  (img, size, extension)
        return record
        
        
    #connect to mysql_pvpsit_table
    def connect_to_pvpsit(self,config_dict):
        #print("eeeeeeee",config_dict['host'])

        mydb = mysql.connector.connect(
            host=config_dict['host'],
            user=config_dict['user'],
            passwd=config_dict['passwd'],
            database=config_dict['database']
        )

        return mydb
    
    
    #insert values #manually
    def insert_image_properties(self,record,mydb):
        mycursor = mydb.cursor()
        sql = "INSERT INTO image_properties (path,size,type)  VALUES( %s, %s ,%s)"
        mycursor.execute(sql, record)
        mydb.commit()
        
        
    #obtain the id of the inserted image
    def retrieve_id(self,img_path,mydb):
        mycursor=mydb.cursor()
        mycursor.execute("select id from image_properties where path = %s",(img_path,))
        id1 = mycursor.fetchone()
        return id1[0]
      
###
    def insert_suggestions(self,suggestions,mydb):
        mycursor = mydb.cursor()
        sql = "INSERT INTO suggestions (color_ub,color_lb,fid)  VALUES( %s, %s ,%s)"
        mycursor.execute(sql, suggestions)
        mydb.commit()


    def face_shape(self,img,file_name,process_path):
        #infile=raw_input("enter name of image:")+('.jpg')
        # initialize dlib's face detector (HOG-based) and then create
        # the facial landmark predictor
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

        # load the input image, resize it, and convert it to grayscale
        image = cv2.imread(img)
        
        #image = imutils.resize(image, width=500)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        frame = image
        # detect faces in the grayscale image
        rects = detector(gray, 1)
        shape_name=process_path+file_name+'_shape.jpg'
        skin_name=process_path+file_name+'_crop.jpg'
        hair_name=process_path+file_name+'_hair.jpg'
        # loop over the face detections
        for (i, rect) in enumerate(rects):
                # determine the facial landmarks for the face region, then
                # convert the facial landmark (x, y)-coordinates to a NumPy
                # array
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)
                # convert dlib's rectangle to a OpenCV-style bounding box
                # [i.e., (x, y, w, h)], then draw the face bounding box
                (x, y, w, h) = face_utils.rect_to_bb(rect)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # show the face number
                skin_tone = frame[y+2:y+h-2, x+2:x+w-2]
                maxi=y+h+5
                #print(maxi)
                hair_color = frame[y-(h/2):y-(h/3) , x+(w/4):x+(3*w/4)]
                #cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                # loop over the (x, y)-coordinates for the facial landmarks
                # and draw them on the image
                #for (x, y) in shape:
                    #cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
        cv2.imwrite(shape_name,image) 
        cv2.imwrite(skin_name,skin_tone)
        cv2.imwrite(hair_name,hair_color)

        #shape=['oval','square','round','oblong','heart','diamond']
        #shape1=random.randrange(len(shape))
        return (skin_name,hair_name,maxi)
    
        
    def color_skin_hair(self,infile1, infile2, numcolors=10, swatchsize=20, resize=150):
        image1 = PIL.Image.open(infile1)
        image1 = Image.Image.resize(image1,(resize, resize),resample=0)
        result1 = Image.Image.convert(image1,'P', palette=PIL.Image.ADAPTIVE, colors=numcolors)
        result1.putalpha(0)
        
        image2 = PIL.Image.open(infile2)
        image2 = Image.Image.resize(image2,(resize, resize),resample=0)
        result2 = Image.Image.convert(image2,'P', palette=PIL.Image.ADAPTIVE, colors=numcolors)
        result2.putalpha(0)

        color1,color2 = self.get_colors(result1,result2,resize)
        hue1 = colorsys.rgb_to_hsv(color1[0],color1[1],color1[2])
        hue2 = colorsys.rgb_to_hsv(color2[0],color2[1],color2[2])
        #print(hue1,hue2) 
        v=hue1[2]
        v_h=hue2[2]
        feat=()
        if(v>210):
            feat=feat+('Fair',)
        elif (v>128 and v<=210):
            feat=feat+('Normal',)
        elif(v<=198):
            feat=feat+('Dark',)        
        
        #print(feat)
        if(v_h<=80):
            feat=feat+('Black',)
        elif(v_h>80 and v_h<=167):
            feat=feat+('Brown',)
        elif(v_h>167):
            feat=feat+('Blonde',)
        return feat
       


    def get_hue_color(self,img,infile1,infile2,resize,numcolors=10):
        image1 = PIL.Image.open(infile1)
        result1 = Image.Image.convert(image1,'P', palette=PIL.Image.ADAPTIVE, colors=numcolors)
        result1.putalpha(0)
        image2 = PIL.Image.open(infile2)
        result2 = Image.Image.convert(image2,'P', palette=PIL.Image.ADAPTIVE, colors=numcolors)
        result2.putalpha(0)
        colors1 = result1.getcolors(resize*resize)
        colors1.sort()
        rgb_1 = colors1[-1][1][0:3]
        
        colors2 = result2.getcolors(resize*resize)
        colors2.sort()
        rgb_2 = colors2[-1][1][0:3]
        
        #print colors1
        #print colors2
        #print rgb_1,rgb_2
        return rgb_1,rgb_2


    def get_colors(self,result1,result2,resize):
        colors1 = result1.getcolors(resize*resize)
        colors1.sort()
        list1=[x[1] for x in colors1]
        face=[x[0] for x in colors1]
        #print(list1,face)
        list2=[x[0]*x[1][0] for x in colors1]
        list3=[x[0]*x[1][1] for x in colors1]
        list4=[x[0]*x[1][2] for x in colors1]
        #print(list2,list3,list4)
        
        r=numpy.sum(list2)/numpy.sum(face)
        b=numpy.sum(list3)/numpy.sum(face)
        g=numpy.sum(list4)/numpy.sum(face)
        
        #print(colors1)
        #print(r,b,g)
        
        colors2 = result2.getcolors(resize*resize)
        colors2.sort()
        
        list5=[x[1] for x in colors2]
        hair=[x[0] for x in colors2]
        list6=[x[0]*x[1][0] for x in colors2]
        list7=[x[0]*x[1][1] for x in colors2]
        list8=[x[0]*x[1][2] for x in colors2]
        
        r_h=numpy.sum(list6)/numpy.sum(hair)
        b_h=numpy.sum(list7)/numpy.sum(hair)
        g_h=numpy.sum(list8)/numpy.sum(hair)
        
        #print(colors2)
        #print(r_h,b_h,g_h)
        return (r,b,g),(r_h,b_h,g_h)
        

    def insert_features(self,features,mydb):
        mycursor = mydb.cursor()
        #print(features) 
        sql = "INSERT INTO features (skin_tone,hair_color,upper_color,lower_color,id)  VALUES(%s ,%s ,%s, %s ,%s)"

        mycursor.execute(sql, features)

        mydb.commit()
        
    #obtain the fid of the inserted image_features
    def retrieve_fid(self,id_,mydb):
        mycursor=mydb.cursor()
        mycursor.execute("select fid from features where id = %s",(id_,))
        id1 = mycursor.fetchone()
        return id1[0]
        
        
#main
if __name__ == '__main__':       

    results,config_dict = parse_initiator()
    if results.bulk_insertion:
        insertion=1
    elif results.single_image:
        insertion=2
    else:
        insertion=3

    if insertion==1:
        dir_path    =   results.bulk_insertion
        img_obj     =   raw_image(dir_path)
        img_obj.extract_properties_bulk_images(config_dict,insertion)
    elif insertion==2:
        dir_path    =   results.single_image
        file_exists(dir_path)
        img_obj     =   raw_image(dir_path)
        img_obj.extract_properties_single_image('',config_dict,insertion)
    elif insertion==3:
        dir_path    =   config_dict['frame']
        img_obj     =   raw_image(dir_path)
        img_obj.capture_frame_from_camera(config_dict)
    else:
        print("--bulk_insertion or --single_image path is not specified.")
        sys.exit(-1)
