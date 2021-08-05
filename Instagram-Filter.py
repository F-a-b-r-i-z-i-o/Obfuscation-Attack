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
