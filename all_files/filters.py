import numpy as np
import PIL 
import noise
import cv2
from scipy.interpolate import UnivariateSpline
from io import BytesIO
from io import StringIO
import base64
import random

from PIL import Image
from PIL import ImageFilter
from PIL import ImageOps
from PIL import ImageEnhance


import sys, os
from os import path
import enum
import time
import argparse
from PIL.ImageChops import overlay


sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__),'NIMA'))

def main(argv):

 
    inputfile = None     #percorso dell'immagine di input
    outputfile = None    #percorso in cui salvare l'immagine di output
    selected_filters = []   #lista dei filtri da applicare all'immagine

    #nomi dei filtri disponibili
    available_filters = ('clarendon','gingham','juno','reyes','lark', 'kelvin', 'lofi', 'hudson', 'aden', '_1977')

    # testi mostrati dalla finestra di aiuto
    input_helptext = "select an input image by name."

    filters_helptext = "select one or more filters. \
        multiple filters will be applied in the order they're written. \
        each filter name can be followed by two optional float values: intensity and alpha. \
        intensity determines in which percentage the filter will be applied to the original image, values 0 - 1 .\
        alpha determines the strengh of the effects applied by the filter. \
        if omitted, any of them will be set to the default value of 1 ."

    output_helptext = "saves the filtered image with the specified name in a specified folder. \
        if the specified folder does not exist, it will be created. \
        if an image with the specified name already exists, it will be overwritten."

    show_helptext = "show the filtered image on screen, pressing any key will close the shown image."

    evaluate_helptext = "evaluate the filtered image with NIMA."

    #parsing parametri linea di comando 
    parser = argparse.ArgumentParser(description='Apply an instagram filter to and image')
    parser.add_argument('--input', '-i', type=str, nargs = 1, default=None, required = True, help=input_helptext)
    #parser.add_argument('--filters', '-f', type=str, nargs = '+', default=None, required = True, help=filters_helptext)
    parser.add_argument('--output', '-o', type=str, nargs = 1, default=None, help=output_helptext)
    parser.add_argument('--show', '-s', action = 'store_true', default= False, help=show_helptext)
    parser.add_argument('--evaluate', '-e', action = 'store_true', default= False, help=evaluate_helptext)

    args = parser.parse_args()
    
    outpath = ""

    if args.output is not None:
        outpath = args.output[0]
    else:
        outpath = ""


    # immagine di input
    if args.input is not None:
        inputfile = args.input[0]
        print(inputfile)
           
    
    if(inputfile is not None):
        #legge l'immagine con opencv 
        image = cv2.imread(inputfile)
        if(image is None):
            print("the specified input image does not exist")
            return
        else:
            #converti l'immagine ad un array di numpy
            image = _cv_to_array(image)
    else:
        print("no input image selected, please select one with -i filename")
        return

    #richiamo randomc salvando dentro filters, al seocndo parametro passo il numero di immagini alle 
    #quali applichero i filtri
    filters = random_c(available_filters, 2)
    
    #inizializzo directori
    imgs_dir = ""
    #scorro la directori andandola a chiamare con i nomi dei filtri "pescati"
    for filter in filters: 
        #salvataggio nome directory con filtro "pescato"
        imgs_dir += filter + "-" 
    
    imgs_dir = imgs_dir[:-1]

    outputFilename = ""

    #scorro il nome dell'immagine all'indietro finche' non trovo la fine '/'
    for i in reversed(args.input[0]):
        if i == "/":
            break
        #salvo il nome del path
        outputFilename = i + outputFilename

    counter = 0
    #for che scorre i filtri della tupla
    for filter in filters:
        #contatore che conta le applicazioni dei filtri alla foto
        counter+=1
        #stampo i filtri casuali presi
        print(filter)
        #randomizzo alpha tra 0.7 e 1 prendendo solo 3 cifre decimali dopo la virgola
        randomA = 1#float("{0:.3f}".format(random.uniform(0.7,1)))
        #randomizzo alpha tra 0.7 e 1 prendendo solo 3 cifre decimali dopo la virgola
        randomI = 1#float("{0:.3f}".format(random.uniform(0.7,1)))
        #applico filtro ad immagine random
        mod_image = apply_filters(image,[(filter,randomA, randomI)])
        
        #controllo che l'immagine con filtro applicata non sia nulla
        if mod_image is None:
            return
        
        #rinomino l'immagine con il nome del filtro e alpha ed intensity settati con valori compresi tra 0 e 1
        outputfile =  outpath + "/" + imgs_dir + "/" + outputFilename[:-5] + "-" + imgs_dir + "  " + " Applicazione: "+str(counter) +" " +  filter.upper() + " alpa: " + str(randomA) + " intensity: " + str(randomI) + " " + outputFilename[-5:] 
        print(outputfile)
        #salvo ogni applicazione del filtro
        save_image(outputfile,mod_image) 
        #applico alla stessa immagine piu' filtri
        image = mod_image
        
    #salvo la sequenza delle applicazioni sui
    save_image(outputfile,mod_image) 


#funzione che applica i filtri alle immagini di input
def apply_filters(mod_image, selected_filters):
        #prende i filtri presenti nella tupla selected filters
        for f in selected_filters:
            print(f)
            if(f[0] == 'clarendon'):
                mod_image = clarendon(mod_image, f[1], f[2])
                
            elif(f[0] == 'gingham'):
                mod_image = gingham(mod_image, f[1], f[2])
                
            elif(f[0] == 'juno'):
                mod_image = juno(mod_image, f[1], f[2])
            
            elif(f[0] == 'reyes'):
                mod_image = reyes(mod_image, f[1], f[2])

            elif(f[0] == 'lark'):
                mod_image = lark_hsv(mod_image, f[1], f[2])
                
            elif(f[0] == 'kelvin'):
                mod_image = kelvin(mod_image, f[1], f[2])
            
            elif(f[0] == 'hudson'):
                mod_image = hudson(mod_image, f[1], f[2])
            
            elif(f[0] == 'lofi'):
                mod_image = lofi(mod_image, f[1], f[2])
            
            elif(f[0] == 'aden'):
                mod_image = aden(mod_image, f[1], f[2])
                
            elif(f[0] == '_1977'):
                mod_image = _1977(mod_image, f[1], f[2])
                
        return mod_image

#funzione che salva immagini
def save_image(outputfile, image):
    #creo directory vuota
    dirname = ''
    #popolo l'output del file
    filename = outputfile
    # separa il nome del file e della directory
    for i in range (len(outputfile), 0, -1):
        if(outputfile[i-1] == "/"):
            filename = outputfile[i:]
            dirname = outputfile[:i-1]
            break
        
    # se la directory non esiste, creala
    if(os.path.isdir(dirname) == False and len(dirname) > 0):
        os.makedirs(dirname)


    # se viene specificata una directory, è necessario aggiungere un "/" 
    if(len(dirname) > 0):
        dirname = dirname + "/"

    #salva l'immagine con opencv
    cv2.imwrite(dirname + "/" + filename, _to_cv_image(image))
    if(len(dirname) > 0):
        print("saved ",filename, " to ",dirname)
    else:
        print("saved ",filename, " to root folder")

#funzione che seleziona n filtri casualmente(senza ripetizione)
def random_c(available_filters, nfilter):
    #controllo che il numero dei filtri applicabili possibili non sia
    #superiore dei filtri disponibili
    if(nfilter >= len(available_filters)):
        return available_filters
    
    
    #creo lista temporanea contenente tupla (avible_filters)
    tmpL = list(available_filters) 
    #inizializzazione random list
    rL = list()
    
    #scorro il numero dei filtri passati in input
    for i in range (0, nfilter):
        #prendo i filtri casulamente dalla lista
        el = random.choice(tmpL)
        #Inserisco i filtri in lista
        rL.append(el)
        #rimuovo i filtri gia' presi in modo da non creare ripetizioni
        tmpL.remove(el)
    return rL
    
def _to_cv_image(image):
    return (image*255.).astype(np.uint8) #[:, :, ::-1] #rgb -> bgr

def _cv_to_array(image):
    return image.astype(np.float32)/255.0 #[:, :, ::1] bgr -> rgb

def _to_pil_image(image):
    if image.shape[2] == 1:
        image = np.resize(image, (*image.shape[:2],))
        return PIL.Image.fromarray((image * 255.0).astype(np.uint8), 'L')
    return PIL.Image.fromarray((image * 255.0).astype(np.uint8))

def _pil_to_array(image):
    if image.mode == 'L':
        npimage = np.array(image).astype(np.float32) / 255.0
        return  np.resize(npimage, (*npimage.shape,1))
    return np.array(image).astype(np.float32) / 255.0

def _cv_to_base64(image):
    #restituisce una matrice a riga singola di tipo CV_8UC1 che contiene un'immagine codificata come matrice di byte.
    buffer = cv2.imencode('.jpg',image)
    #conversione di dati binari in formato a 64 bit 
    return base64.b64encode(buffer[1]).decode()

def _file_to_cv(image):
    # read as bytes
    image = image.read()
    # convert byte image image to numpy array
    image = np.frombuffer(image, np.uint8)
    # decode bytes as a cv image
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image

def _file_to_array(image):
    # read as bytes
    image = image.read()
    # convert byte image image to numpy array
    image = np.frombuffer(image, np.uint8)
    # decode bytes as a cv image
    image = cv2.imdecode(np, cv2.IMREAD_COLOR)
    return _cv_to_array(image)

def _perlin_array(shape = (200, 200),
                  scale = 10.,
                  octaves = 2, 
                  persistence = 0.5, 
                  lacunarity = 2.0, 
                  seed = 12345):
    arr = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            arr[i][j] = noise.pnoise2(float(i) / scale, 
                                      float(j) / scale,
                                      octaves=octaves,
                                      persistence=persistence,
                                      lacunarity=lacunarity)
    max_arr = np.max(arr)
    min_arr = np.min(arr)
    norm_me = lambda x: (x-min_arr)/(max_arr - min_arr)
    norm_me = np.vectorize(norm_me)
    arr = norm_me(arr)
    return arr

def show_image(image):
    img = _to_pil_image(image)
    img.show()

def scale(image, factor=2.0):
    pil_image = _to_pil_image(image)
    target_size = pil_image.size
    pil_image = pil_image.resize((int(pil_image.width * factor), int(pil_image.height * factor)))
    left = (pil_image.size[0] - target_size[0])/2
    top = (pil_image.size[1] - target_size[1])/2
    right = (pil_image.size[0] + target_size[0])/2
    bottom = (pil_image.size[1] + target_size[1])/2
    pil_image = pil_image.crop((left, top, right, bottom))
    return _pil_to_array(pil_image)

def rotate(image, angle = 0.0):

    pil_image = _to_pil_image(image)
    pil_image = pil_image.rotate(angle)
    return _pil_to_array(pil_image)

# effetto di vignettatura nera sui bordi dell'immagine
def vintage(image, factor = 1.0):
    im = _to_cv_image(image)
    rows, cols = im.shape[:2]# Create a Gaussian filter
    kernel_x = cv2.getGaussianKernel(cols,200)
    kernel_y = cv2.getGaussianKernel(rows,200)
    kernel = kernel_y * kernel_x.T
    _filter = (255 * kernel / np.linalg.norm(kernel)) * factor
    im = _cv_to_array(im)
    for i in range(3):
        im[:,:,i] *= _filter 
    return im

def sharpen(image):
    pil_image = _to_pil_image(image)
    mod_image = pil_image.filter(PIL.ImageFilter.SHARPEN)
    return _pil_to_array(mod_image)

def smooth_more(image):
    pil_image = _to_pil_image(image)
    mod_image = pil_image.filter(PIL.ImageFilter.SMOOTH_MORE)
    return _pil_to_array(mod_image)

def gaussian_blur(image, radius=2):
    pil_image = _to_pil_image(image)
    mod_image = pil_image.filter(PIL.ImageFilter.GaussianBlur(radius=radius))
    return _pil_to_array(mod_image)

def jpeg_compression(image, quality = 90):
    pil_image = _to_pil_image(image)  
    buffer = BytesIO()
    pil_image.save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    pil_image = PIL.Image.open(buffer)
    return _pil_to_array(pil_image)
 
def perlin_noise(image, octaves = 6, 

                        scale = 10, 
                        alpha = 0.1):
    c_img = image.copy()
    p_nois = _perlin_array(shape=(c_img.shape[0], c_img.shape[1]),
                           octaves=octaves,
                           scale=scale)
    for c in range(image.shape[2]):
        c_img[:,:,c] += (p_nois * alpha)
    c_img /= c_img.max()
    return c_img 

def replace_color(image,hl=0,sl=0,vl=0,hu=0,su=0,vu=0,nred=0,ngreen=0,nblue=0):
	rows, cols = image.shape[:2]
	# Convert BGR to HSV
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	# define range of color in HSV
	lower = np.array([hl,sl,hu])
	upper = np.array([hu,su,vu])
	# Threshold the HSV image to get only blue colors
	color = cv2.inRange(hsv, lower, upper) 
	# Replace color
	image[color>0]=(nblue,ngreen,nred)
	return image

# correzione gamma, il valore di default non cambia l'immagine
def gamma_correction(image, gamma=2.2):
  invgamma = 1.0 / gamma
  imin, imax = image.min(), image.max()
  newimg = image.copy()
  newimg = ((newimg - imin) / (imax - imin)) ** invgamma
  newimg = newimg * (imax - imin) + imin
  return newimg

# modifica saturazione, il valore di default non cambia l'immagine
def edge_enhance(image, alpha=0.5):
    pil_image = _to_pil_image(image)
    converter = PIL.ImageEnhance.Color(pil_image)
    pil_image = converter.enhance(alpha)
    return _pil_to_array(pil_image)

# modifica luminosità, il valore di default non cambia l'immagine
def brightness (image, alpha = 1.0):
    pil_image = _to_pil_image(image)
    converter = PIL.ImageEnhance.Brightness(pil_image)
    pil_image = converter.enhance(alpha)
    return _pil_to_array(pil_image)

# modifica contrasto, il valore di default non cambia l'immagine
def contrast(image, alpha=1.0):
    pil_image = _to_pil_image(image)
    converter = PIL.ImageEnhance.Contrast(pil_image)
    pil_image = converter.enhance(alpha)
    return _pil_to_array(pil_image)

# enfatizza i contorni per alpha > 1, sfuma l'immagine per alpha < 1
def sharpness (image, alpha = 1.0):
    pil_image = _to_pil_image(image)
    converter = PIL.ImageEnhance.Sharpness(pil_image)
    pil_image = converter.enhance(alpha)
    return _pil_to_array(pil_image)

#calculate a lookup table through a spline function
def spread_lookup_table(x, y):
    spline = UnivariateSpline(x, y)
    return spline(range(256))

# moltiplica i valori dei canali dell'immagine per i rispettivi parametri passati
# aumentare rosso e diminuire blu per ottenere colori più caldi
# aumentare blu e diminuire rosso per ottenere colori più freddi
def hue(image, red = 1, green = 1, blue = 1):
    
     #converts pixel values from 0-1 to 0-255
    image = _to_cv_image(image)

    # lists of 4 values used to construct the lookup tables
    base = np.array([0, 64, 128, 255]) 
    redValues = np.clip(base * red, 0, 255)
    greenValues = np.clip(base * green, 0, 255)
    blueValues = np.clip(base * blue, 0, 255)
    
    # Make sure that the last value remains 255, so whites don't get modified
    redValues[-1] = \
    greenValues[-1] = \
    blueValues[-1] = 255

    # calculate a lookup table of length 256, starting from the base array
    # the lookup tables will contain the new channel values
    redLookupTable = spread_lookup_table(base, redValues)
    greenLookupTable = spread_lookup_table(base, greenValues)
    blueLookupTable = spread_lookup_table(base, blueValues)

    # opencv uses a BGR format
    # LUT transforms an array based on a lookup table:
    # every old value is swapped with its mapped value in the lookup table
    red_channel = cv2.LUT(image[:,:, 2], redLookupTable)
    green_channel = cv2.LUT(image[:,:, 1], greenLookupTable)
    blue_channel = cv2.LUT(image[:,:, 0], blueLookupTable)

    # ensure that channels don't overflow
    np.putmask(red_channel, red_channel > 255, 255)
    np.putmask(green_channel, green_channel > 255, 255)
    np.putmask(blue_channel, blue_channel > 255, 255)

    #assegna new channels to the image
    image[:,:, 0] = blue_channel
    image[:,:, 1] = green_channel
    image[:,:, 2] = red_channel 

    np_image = _cv_to_array(image)

    return np_image

#calcola le lookuptable in hue utilizzando una spline
def spreadLookupTable(x, y):
    spline = UnivariateSpline(x, y)
    return spline(range(256))


# aggiunge un colore piatto a tutta l'immagine, alpha ne indica il livello di trasparenza:
# con 0 si ottiene l'immagine originale, con 1 un colore opaco
def vignette(image, alpha = 0, red = 255, green = 255, blue = 255):
    from PIL import ImageDraw

    if( alpha < 0 or alpha > 1):
        raise RuntimeError("invalid alpha for vignette effect")

    #disegna un rettangolo trasparente del colore specificato
    alpha = int(alpha * 255)
    transparent = Image.new('RGBA', (image.shape[1],image.shape[0]), (blue, green, red, alpha))

    #converte immagine di partenza a pillow
    output = _to_pil_image(image)

    # applica il rettangono all'immagine di partenza
    output.paste(transparent, (0,0), transparent)

    return _pil_to_array(output)


#operazioni di thresholding
#ricevono in input il range di colori da selezionare

# lower e upper bound specificano il range di triple HSV dal quale selezionare i pixel dell'immagine
# color prende argomenti della classe Color_HSV, se assegnato sovrascrive lower_bound e upper_bound  
def select_by_hsv(image, lower_bound = (90, 50, 30), upper_bound = (130,255,230), color = None):
    
    if(color is not None):
        lower_bound = (color.value[0], 0, 0)
        upper_bound = (color.value[1], 255, 255)
    
    image = _to_cv_image(image)
    temp = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_bound = np.array( lower_bound , dtype = np.uint8, ndmin = 1)
    upper_bound = np.array( upper_bound , dtype = np.uint8, ndmin = 1)

    #ricava la maschera di pixel selezionati. la maschera sarà in scala di grigio, con valore 0(nero) sui pixel scartati e 255(bianco) su quelli selezionati
    mask = cv2.inRange(temp, lower_bound , upper_bound)

    #la maschera va convertita a bgr per poterle riapplicare il colore
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    #assegna i colori originali ai pixel selezionati
    temp = _cv_to_array(image & mask_bgr)

    return temp


# sostituzione: sostituisce i pixel di im2 != (0, 0, 0) ai pixel di im1
def img_replace(im1, im2):
    im1 = np.where(im2 == 0, im1 , im2)
    return im1

# intersezione: ritorna i pixel di im1 che sono presenti in entrambe le immagini
def img_intersection(im1, im2):
    im1 = np.where(im2 != 0, im1 , 0)
    return im1

# unisce i pixel di entrambe le immagini. Per i pixel presenti in entrambi, quelli di im2 sovrascriveranno quelli di im1
def img_union(im1, im2):
    im1 = np.where(im2 != 0, im2 , im1)

# fonde le immagini in percentuale data dal parametro alpha, riferito alla seconda immagine
def interpolate(im1, im2, alpha):
    return ((1.0 - alpha) * im1 + alpha * im2)


#filtri di instagram
def clarendon(in_image, intensity = 1, alpha = 1):
    image = contrast(in_image, 1.2 * alpha)
    image = edge_enhance(image, 2.0 * alpha)
    image = hue(image, 0.6 * alpha, 1.0 * alpha, 1.2 * alpha)
    out_image = interpolate(in_image, image, intensity)

    return out_image
        
def gingham(in_image, intensity = 1 , alpha = 1):
    image = brightness(in_image, 1.1 * alpha)
    image = edge_enhance(image, 1.1 * alpha)
    image = contrast(image, 0.7 * alpha)
    out_image = interpolate(in_image, image, intensity)

    return out_image

def juno(in_image, intensity = 1, alpha = 1):
    image = contrast(in_image, 1.15 * alpha)
    image = edge_enhance(image, 1.1 * alpha)
    image = gamma_correction(image, 1.3 * alpha)
    out_image = interpolate(in_image, image, intensity)

    return out_image

def reyes(in_image, intensity = 1 ,alpha = 1):
    image = contrast(in_image, 0.9 * alpha)
    image = edge_enhance(image, 0.75 * alpha)
    image = brightness(image, 1.2 * alpha)
    image = gamma_correction(image, 1.2 * alpha)
    image = hue(image, 1.1 * alpha, 1.1 * alpha, 1 * alpha)
    out_image = interpolate(in_image, image, intensity)

    return out_image

def lark_hsv(in_image, intensity = 1, alpha = 1):
    image = gamma_correction(in_image, 0.8 * alpha)

    #seleziona il blu. la maschera deve solo selezionare i pixel da modificare e non deve essere modificata dall'alpha
    mask = gamma_correction(image, 1.3)
    mask = select_by_hsv(mask, lower_bound = (90, 50, 30), upper_bound = (130,255,230))

    #applica filtro ai pixel selezionati
    filtered = hue(mask, 0.8 * alpha, 0.8 * alpha, 1.2 * alpha)
    filtered = img_intersection(filtered, mask)
    image = img_replace(image, filtered)
    out_image = interpolate(in_image, image, intensity)

    return out_image


    #modifico contrasto all'immagine in input e moltiplico per alfa
    image = contrast(in_image,   1.2 * alpha)
    #modifico la saturazione dell'immagine anch'essa moltilpicata per alfa (2 immagine)
    image = edge_enhance(image, 1.5 * alpha)
    #modifico la luminosita' e moltiplico per alfa (2 immagine)
    image = brightness(image,  1.0 * alpha)
    #vado a correggere l'immagine 
    image = gamma_correction(image, 1.1 * alpha)
    #modifico i colori dell'immagine in base ad i valori passati e moltiplico per alfa
    image = hue(image, 1.2 * alpha, 1.0 * alpha, 1.1 * alpha)
    
   
  
 
    
    #creo l'immagine di out fonde le immagini in percentuale data dal parametro alpha, riferito alla seconda immagine
    out_image = interpolate(in_image, image, intensity)

   
    return out_image

def kelvin(in_image, intensity=1, alpha=1):
    #modifico contrasto all'immagine in input e moltiplico per alfa
    image = contrast(in_image, 1.0 * alpha)
    #modifico la saturazione dell'immagine anch'essa moltilpicata per alfa (2 immagine)
    image = edge_enhance(image, 1.0 * alpha)
    #modifico la luminosita' e moltiplico per alfa (2 immagine)
    image = brightness(image, 1.1 * alpha)
    
  
    #vado ad applicare il filtro "vignette" alla prima immagine di input andando a modificare i valor rgb e moltiplicando per alpha
    image = vignette(image, 0.2  , 220 * int(alpha) , 97 * int(alpha)  , 113 * int(alpha))
    

    #creo l'immagine di out fonde le immagini in percentuale data dal parametro alpha, riferito alla seconda immagine
    out_image = interpolate(in_image, image, intensity)
    return out_image

def _1977(in_image, intensity=1, alpha=1):
    #modifico contrasto all'immagine in input e moltiplico per alfa
    image = contrast(in_image, 1.3 * alpha)
    #modifico la saturazione dell'immagine anch'essa moltilpicata per alfa (2 immagine)
    image = edge_enhance(image, 1.3 * alpha)
    #modifico la luminosita' e moltiplico per alfa
    image = brightness(image, 0.9 * alpha)
    #vado a correggere l'immagine 
    image = gamma_correction(image, 1.2 * alpha)
    #vado ad applicare il filtro "vignette" alla prima immagine di input andando a modificare i valor rgb e alpha
    image = vignette(image, 0.3 , 243  * int(alpha), 106 * int(alpha), 188 * int(alpha))
    
    #creo l'immagine di out fonde le immagini in percentuale data dal parametro alpha, riferito alla seconda immagine
    out_image = interpolate(in_image, image, intensity)
    return out_image

def lofi(in_image, intensity=1, alpha=1):
    #modifico contrasto all'immagine in input e moltiplico per alfa
    image = contrast(in_image, 1.5 * alpha)
    #modifico la saturazione dell'immagine anch'essa moltilpicata per alfa (2 immagine)
    image = edge_enhance(image, 1.4 * alpha)
    #modifico la luminosita' e moltiplico per alfa 
    image = brightness(image, 0.9 * alpha)
    
    #creo l'immagine di out fonde le immagini in percentuale data dal parametro alpha, riferito alla seconda immagine
    out_image = interpolate(in_image, image, intensity)
    return out_image

def aden(in_image, intensity = 1 ,alpha = 1):
   
    #utilizzo funzione rotate per ruotare imaggine input -20 
    image = rotate(in_image, -20)
    #modifico il contrasto dell'immagine in input
    image = contrast(in_image, 0.95 * alpha)
    #modifico saturazione della 2 immagine
    image = edge_enhance(image, 0.85 * alpha)
    #modifico luminosita' seconda immagine
    image = brightness(image, 1.0 * alpha)
    #moltiplica i valori dei canali dell'immagine per i rispettivi parametri passati
    image = hue(image, 1.1 * alpha, 1.0 * alpha, 1.0 * alpha)
    
    #creo l'immagine di out fonde le immagini in percentuale data dal parametro alpha, riferito alla seconda immagine
    out_image = interpolate(in_image, image, intensity)
    return out_image

def hudson(in_image, intensity=1, alpha= 1):
    
    #modifico il contrasto dell'immagine in input
    image = contrast(in_image, 1.3 * alpha)
    #modifico luminosita' 2 immagine
    image = brightness(image, 1.0 * alpha)
    #modifico saturazione della 2 immagine
    image = edge_enhance(image, 1.1 * alpha)
    #vado a modificare i canali rgb in base ad i parametri passati e moltiplico per alpha
    image = hue(image, 1.2 * alpha, 1.0 * alpha, 1.2 * alpha)
  

    #creo l'immagine di out fonde le immagini in percentuale data dal parametro alpha, riferito alla seconda immagine
    out_image = interpolate(in_image, image, intensity)

    return out_image

# definisce range HSV predefiniti per alcuni colori
class Color_HSV(enum.Enum):
    red = (0, 10)
    orange = (10, 20)
    yellow = (20, 35)
    green = (35, 75)
    lightblue = (75, 100)
    blue = (105, 125)
    purple = (130, 150)

if __name__ == '__main__':
    main(sys.argv[1:])

