# Plates-Recognition
Project for Image Processing Course

link to github: https://github.com/JinLobana/Plates-Recognition

Command to run project:  
python3 Lubina_Jan.py dane/train_1 Lubina_Jan.json

# Krótki opis kodu:
## Operacje na obrazie:
### Preprocessing
Jedna funkcja preprocessing image, wczytuje obrazy z folderu, wstępnie przerabia, zapisuje do odpowiednich list.  
### Detekcja tablic
Następnie pracowałem tylko na szarym obrazie, nie zdążyłem porównać efektów z maskami z użyciem obrazów kolorowych w reprezentacji HSV. Przez brak czasu w funkcji detecting_plate() znajduje się obróbka obrazów w szkali szarości. Robię progowanie, filtr medianowy, otwarcie. Następnie używam mojej ulubionej funkcji w OpenCV *connectedComponentsWithStats*. Na podstawie statystyk zwracanych przez nią znajduję rejestrację. Następnie znajduję kontur, aproksymuję go funkcją Poly, wyszukuje cztery rogi, przekształcam oryginalny szary obraz perspektywicznym przekształceniem. Zwracam samą wykrytą rejestrację po przkształceniu. 

### Wyodrębnianie region of interst dla każdego znaku
```isolating_letters_from_plate``` Ponownie proguje, filtruje etc. Dodaje dodatkowe poksele z każdej strony obrazu, w celu lepszego wykrywania znaków. Tworzy negatyw obrazu binarnego. Ponownie *connectedComponentsWithStats* \<3. Odpowiednimi warunkami znajduję znaki, tworzę prostokąty, które ekstraktuję z obrazu. Sortuję je i zwracam. 

## Operacje OCR oparte na template matching

Wykonałem trzy metody, niestety żadna nie okazała się wystarczająco skuteczna. Prawdopodobnie można pobawić się jeszcze filtracją obrazu, znalezieniem odpowiednich paramtrów wszystkich funkcji etc., ale niestety nie było już na to czasu. 

### ```feature_descriptor```
> Tworzy deskryptor SIFT. dostaje jeden znak do rozpoznania i liste szablonów odpowiednio przygotowanych obrazów w czcionce, jaka jest używana na polskich tablicach rejestracyjnych. Obrazy te tworzy skrypt ```scripts/generating_dataset.py```. Tworzy matcher, używa knn match, liczy dystans za pomocą testu Lowe'a. Najlepszy template jest zwracany.  
> Problemem jest odwracanie (np. 7 są wykrywane jako L), ale nie idzie dobrać skalującego deskryptora, który jednocześnie nie uwzględnia rotacji szblonów. 

### ```template_matching```
Najprostszy template matching jaki istnieje. Beznadzijne wyniki dostawałem niestety. *minMaxLoc* funkcja używana do obliczania najlepszego dopasowania. 

### ```shapes_matching```
Bardzo ciekawa i prosta metoda, dawała mi koniec końców najlepsze rezultaty. Tworzę kontury dla każdego obiektu i szablonu, wykorzystuję funkcję *matchShapes*, której podaje się dwa kontury i zwraca jak bardzo są do siebie podobne (im mniejsza wartość tym lepiej). 

## ```main```

O zmiennej *cheating* opowiem na końcu ;p . Tworzę argumenty do wywołania pliku, globalne zmienne, głównie listy, następnie wczytuję szablony, robię preprocessing.  
Dla każdego z obrazów:
- Wykrywam tablice i znaki
- W pętli dla każdego znaku  
    - Robie feature descriptor i shape matching
    - Zapisuje do stringów odczytane wartości
- Printuję, zapisuje do słownika i dalej do pliku

