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
