# Numele proiectului: Movie Recommendations
  
## Prezentare generală  
Sistemul de recomandare a filmelor este conceput pentru a aborda problema găsirii filmelor potrivite dintr-o cantitate enorma de date. Scopul proiectului este de a reduce la minimum timpul pe care utilizatorii îl petrec în căutarea de filme prin furnizarea de recomandări personalizate.  
  
- Scop:   
Scopul sistemului este simplificarera procesului de selecție a filmelor, utilizând algoritmi care analizează istoricul de vizionare și preferințele utilizatorilor pentru a sugera filme relevante.  
  
- Obiective:   
Dezvoltarea unui algoritm care poate prezice cu precizie preferințele utilizatorilor pe baza istoricului de vizionare. 
Crearea unei interfețe simple și intuitive pentru ca utilizatorii să interacționeze cu sistemul.  
Îmbunătățirea continuă a acurateței recomandărilor prin încorporarea feedback-ului utilizatorilor și adaptarea la noile date despre filme.  
  
## Table of Contents  
1.  [Introducere](#introduction)  
2. [Models Overview](#models-overview)  
3. [Installation and Setup](#installation-and-setup)  
4. [Usage](#usage)  
5. [Detalii despre model](#model-details)  
6. [Training and Evaluation](#training-and-evaluation)  
7. [Results and Discussion](#results-and-discussion)  
8. [Contribuții](#contributions)  
9. [License](#license)  
10. [Acknowledgments](#acknowledgments)  
11. [References](#references)    
## Introduction  
Proiectul se concentrează pe dezvoltarea unui sistem de recomandare a filmelor folosind algoritmi de învățare automată pentru a analiza și prezice preferințele utilizatorilor pe baza istoricului de vizionare. Acesta utilizează tehnici de filtrare colaborativă pentru a identifica filmele care corespund gusturilor individuale, simplificând astfel procesul de selecție a filmelor pe platformele de streaming. 
  
## Models-overview

- Numele modelului: Recomandare de filme prin diferență temporală (TD)  
  - Scop/caz de utilizare: Acest model este conceput pentru a recomanda filme utilizatorilor pe baza istoricului de vizionare al acestora. Este deosebit de potrivit pentru cazurile în care se așteaptă ca preferințele utilizatorilor să evolueze în timp, deoarece se adaptează la schimbările de comportament și de evaluări ale utilizatorilor.  
  - Principiul de lucru de bază:  
    - TD Movie Recommender este construit folosind Python, cu accent pe manipularea datelor folosind Pandas și Numpy.  
    - Acesta citește date despre filme și evaluări și utilizează o abordare de învățare prin diferență temporală (TD), un tip de tehnică de învățare prin consolidare.  
    - Modelul păstrează valori Q (valori de calitate) pentru fiecare film, reprezentând utilitatea așteptată a recomandării filmului respectiv.  
    - Sistemul actualizează aceste valori Q pe măsură ce procesează istoricul de evaluare a filmelor de către utilizator, învățând să prezică preferințele utilizatorului.  
    - Procesul de recomandare presupune selectarea filmelor cu cele mai mari valori Q pe baza stării curente (ultimul film evaluat de utilizator) și a unui factor de explorare (epsilon).  
    - Funcția update_Q_values ajustează valorile Q folosind o rată de învățare (alpha) și un factor de actualizare (gamma), echilibrând recompensele imediate (ratingurile actuale) cu recompensele viitoare (ratingurile așteptate).  
    - Modelul urmărește să ofere o listă de top N de recomandări de filme, adaptată dinamic la gusturile în evoluție ale utilizatorului.  
  

- Numele modelului: SARSA-Based Movie Recommender  
  - Scop/caz de utilizare: Acest model este conceput pentru a oferi recomandări personalizate de filme prin învățarea din interacțiunile utilizatorului într-o manieră dinamică și adaptivă. Este potrivit pentru mediile în care preferințele utilizatorilor nu sunt statice și se pot schimba în timp.  
  - Principiul de lucru de bază:  
  
    - Recomandatorul de filme bazat pe SARSA este dezvoltat folosind Python și utilizează în principal Pandas și Numpy pentru manipularea datelor și calcule.  
    - Modelul citește și procesează datele de evaluare a filmelor și a utilizatorilor, aplicând algoritmul de învățare SARSA (State-Action-Reward-State-Action), o variantă a învățării prin întărire.  
    - În acest model, valorile Q (valori de calitate) sunt menținute pentru perechi de stări (filmul curent) și acțiuni (următorul film), reprezentând utilitatea așteptată a recomandării unei secvențe specifice de filme.  
    - Modelul actualizează aceste valori Q luând în considerare starea și acțiunea curente, recompensa (evaluarea utilizatorului), precum și starea și acțiunea următoare. Acest proces este guvernat de parametri precum rata de învățare (alfa) și factorul de actualizare (gamma).  
    - O caracteristică esențială a SARSA este faptul că își actualizează valorile Q nu doar pe baza stării și acțiunii curente, ci și pe baza stării următoare și a acțiunii alese în acea stare, ceea ce face din el un algoritm de învățare în funcție de politică. - The recommendation process involves selecting movies with the highest Q-values, and the system uses an exploration factor (epsilon) to balance between exploiting known information and exploring new possibilities.  
    - Procesul de recomandare implică selectarea filmelor cu cele mai mari valori Q, iar sistemul utilizează un factor de explorare (epsilon) pentru a echilibra între exploatarea informațiilor cunoscute și explorarea de noi posibilități.  
    - Obiectivul modelului este de a furniza o listă de recomandări de filme de top-n pentru un utilizator, derivate din învățarea tiparelor de vizionare ale acestuia și din adaptarea la preferințele sale în schimbare.  
  
- Modelul 3: Period Genre-Based Movie Recommender
  - Scop/Caz de Utilizare:
    - Acest model este destinat recomandării filmelor pe baza perioadelor istorice și genurilor. Este util pentru a sugera filme utilizatorilor bazându-se pe preferințele lor în evoluție.
  - Principiul de Funcționare:
    - Modelul îmbină date despre filme și evaluările utilizatorilor pentru a învăța preferințele acestora. Utilizează Q-learning, unde valori Q sunt asociate cu combinații de perioade și genuri de filme. Se antrenează prin actualizarea acestor valori Q bazându-se pe istoricul de evaluări al utilizatorilor, recomandând filme cu cele mai mari valori Q care nu au fost încă vizionate de utilizator. 
  
## Differences  
| Caracteristică/Model                      | TD Movie Recommender                         | Recomandare de Filme Bazată pe SARSA           | Period Genre-Based Recommender |
|-------------------------------------------|---------------------------------------------|-----------------------------------------------|--------------------------------|
| **Abordare de Învățare**                  | Învățare prin Diferență Temporală (TD)      | Învățare prin Stare-Actiune-Reward-Stare-Actiune (SARSA) | Învățare bazată pe Q-values pentru perioade și genuri |
| **Avantaje**                              |                                             |                                               |                                |
| *Adaptabilitate*                          | Se adaptează la preferințele utilizatorului în timp. | Se adaptează continuu la schimbările comportamentului utilizatorului. | Adaptabil la preferințe pe baza perioadelor și genurilor |
| *Complexitatea Algoritmului*              | Relativ simplu și ușor de implementat.    | Gestionează secvențe de acțiuni, ceea ce poate duce la recomandări mai nuanțate. | Moderată; combină analiza perioadelor și genurilor |
| *Precizia Predictivă*                     | Eficient în scenarii cu preferințe stabile. | Potențial mai precisă, deoarece ia în considerare secvențe de acțiuni și recompense. | Poate oferi recomandări mai specifice bazate pe perioade și genuri |
| *Explorare vs. Exploatare*                | Menține un echilibru între explorarea filmelor noi și exploatarea preferințelor cunoscute. | Menține un echilibru explicit între explorare și exploatare prin învățarea politicilor. | Echilibrează explorarea și exploatarea prin selecția bazată pe gen și perioadă |
| **Dezavantaje**                           |                                             |                                               |                                |
| *Sensibilitate la Datele Inițiale*        | Recomandările inițiale pot fi mai puțin precise până când se adună suficiente date. | Este posibil să fie nevoie de mai multe date pentru a învăța eficient secvențe de acțiune. | Necesită date inițiale diverse pentru a acoperi diferite perioade și genuri |
| *Receptivitate la Schimbare*              | S-ar putea să se adapteze mai greu la schimbările bruște ale preferințelor utilizatorilor. | Mai receptivă la schimbările imediate în comportamentul utilizatorului datorită învățării pe bază de politici. | Se adaptează la schimbările în preferințele de gen și perioadă ale utilizatorilor |
| *Complexitate Computațională*             | Complexitate computațională mai mică.    | Complexitate de calcul mai mare datorită luării în considerare a perechilor stare-acțiune. | Complexitate moderată, influențată de diversitatea genurilor și perioadelor |
| *Potrivire pentru Cazurile de Utilizare*  | Mai potrivită pentru mediile stabile în care preferințele utilizatorilor nu se schimbă rapid. | Ideală pentru mediile dinamice în care preferințele utilizatorilor se schimbă frecvent. | Potrivită pentru preferințele legate de genuri și perioade specifice |

  
## Instalare și configurare  
- Instrucțiuni detaliate privind modul de configurare a proiectului. Includeți:  
  - Condiții prealabile: python-3.12  
  - Configurați mediul virtual  
    - Navigați în directorul proiectului și rulați `python -m venv venv`.  
    - Activați mediul virtual:  
      - Pe Windows, utilizați venv\Scripts\activate.  
      - Pe MacOS/Linux, utilizați sursa venv/bin/activate.  
  - Instalați bibliotecile și dependențele  
    - pandas  
    - numpy  
  - Clonați proiectul:  
    - https://github.com/anastasiasandu/ReinforcementLearning.git  

  
## Usage  
  - Model 1: `python approach1.py`  
  - Model 2: `python approach2.py`  
  - Model 3: `python approach3.py`
  
## Model Details  
## Model Details  
| Caracteristică                   | TD Movie Recommender                              | Recomandare de Filme Bazată pe SARSA                   | Period Genre-Based Recommender |
|---------------------------------|----------------------------------------------------|-------------------------------------------------------|--------------------------------|
| **Arhitectură și Design**       | - Bazată pe învățarea prin Diferență Temporală (TD). | - Folosește algoritmul SARSA (Stare-Actiune-Recompensă-Stare-Actiune). | - Se bazează pe analiza perioadelor istorice și genurilor filmelor. |
|                                 | - Menține valorile Q pentru fiecare film.           | - Menține valorile Q pentru perechi stare-acțiune (secvențe de filme). | - Menține valorile Q pentru combinații de perioade și genuri. |
|                                 | - Actualizează valorile Q utilizând evaluările utilizatorilor ca recompense. | - Actualizează valorile Q pe baza tranzițiilor între perechile stare-acțiune. | - Actualizează valorile Q pe baza preferințelor utilizatorilor pentru perioade și genuri. |
| **Formatul și Cerințele Datelor de Intrare** | - Necesită fișierele `movies.dat` și `ratings.dat`. | - Necesită, de asemenea, fișierele `movies.dat` și `ratings.dat`. | - Necesită fișierele `movies.dat` și `ratings.dat`, cu procesare suplimentară pentru perioade și genuri. |
|                                 | - Datele sunt în format tabular, codate în 'latin-1'.  | - Datele au un format tabular similar, codate în 'latin-1'. | - Datele sunt prelucrate pentru a identifica perioade istorice și genuri. |
|                                 | - Datele cheie: MovieID, Titlu, Genuri pentru filme; UserID, MovieID, Evaluare pentru evaluări. | - La fel ca în cazul TD Movie Recommender. | - Include analiza anilor și genurilor pentru filme, pe lângă datele de bază. |
| **Formatul și Interpretarea Datelor de Ieșire**  | - Oferă o listă de titluri de filme ca recomandări. | - Oferă o listă de titluri de filme ca recomandări. | - Oferă recomandări bazate pe cele mai bune combinații de perioade și genuri. |
|                                 | - Recomandările se bazează pe cele mai mari valori Q. | - Recomandările sunt derivate din valorile Q ale perechilor stare-acțiune. | - Recomandările sunt selectate pe baza valorilor Q pentru perioade și genuri preferate. |

## Training and Evaluation  
| Proces de Antrenament și Evaluare             | Recomandare de Filme Bazată pe Diferență Temporală (TD) | Recomandare de Filme Bazată pe SARSA        | Period Genre-Based Recommender |
|----------------------------------------------|--------------------------------------------------------|-------------------------------------------|--------------------------------|
| **Pregătirea Datelor și Prelucrarea**        | - Datele sunt încărcate din fișierele `movies.dat` și `ratings.dat`. | - Datele sunt încărcate din fișierele `movies.dat` și `ratings.dat`. | - Datele sunt încărcate din `movies.dat` și `ratings.dat`. Prelucrarea suplimentară pentru perioade și genuri. |
|                                              | - Datele sunt filtrate pentru a selecta coloanele relevante (UserID, MovieID, Rating). | - Datele sunt filtrate pentru a selecta coloanele relevante (UserID, MovieID, Rating). | - Filtrare pentru UserID, MovieID, Rating. Codificarea 'latin-1'. |
|                                              | - Se utilizează codificarea 'latin-1' din cauza formatului datelor. | - Se utilizează codificarea 'latin-1' din cauza formatului datelor. | - Evaluările utilizatorilor ca recompense. Analiza anilor și genurilor pentru filme. |
| **Procedura de Antrenament**                  | - Modelul utilizează învățarea prin Diferență Temporală (TD). | - Modelul utilizează algoritmul SARSA (Stare-Actiune-Recompensă-Stare-Actiune) pentru învățare. | - Aplică Q-learning pentru perioade și genuri. Inițializează și actualizează valorile Q. |
|                                              | - Valorile Q pentru fiecare film sunt inițializate aleator. | - Valorile Q pentru perechi stare-acțiune sunt inițializate aleator. | - Parametri: Rată de Învățare (alpha), discount (gamma). |
| **Metrici și Metode de Evaluare**            | - Evaluarea este adesea implicită, concentrată pe capacitatea de a se adapta la preferințele utilizatorului în timp. | - Evaluarea este adesea implicită, cu accent pe capacitatea modelului de a se adapta la preferințele dinamice ale utilizatorului. | - Evaluarea implicită, axată pe adaptarea la preferințele utilizatorului. |
|                                              | - Recomandările sunt generate pe baza valorilor Q învățate. | - Recomandările sunt generate pe baza valorilor Q ale perechilor stare-acțiune. | - Recomandări bazate pe valorile Q pentru perioade și genuri. Nu există metrici de evaluare explicite; feedback-ul utilizatorului esențial. |
|                                              | - Nu există metrici de evaluare explicite; feedbackul utilizatorului poate ghida rafinarea modelului. | - Nu există metrici de evaluare explicite; feedbackul utilizatorului poate ghida rafinarea modelului. | - Satisfacția utilizatorului și interacțiunea ca indicatori de succes. |

## Results and Discussion  
| Aspect                                      | Recomandare de Filme Bazată pe Diferență Temporală (TD) | Recomandare de Filme Bazată pe SARSA        | Period Genre-Based Recommender |
|---------------------------------------------|--------------------------------------------------------|-------------------------------------------|--------------------------------|
| **Metrici de Performanță**                   | Implicit, satisfacția utilizatorului și interacțiunea  | Implicit, satisfacția utilizatorului și interacțiunea | Implicit, concentrare pe preferințele pe baza perioadelor și genurilor |
| **Focus**                                   | Recomandări individuale pentru filme                   | Recomandări sub forma secvențelor de filme | Recomandări bazate pe combinații de perioade istorice și genuri |
| **Viteză de Adaptare**                      | Adaptare rapidă la preferințele care se schimbă rapid  | Eficient în modelarea preferințelor dinamice | Adaptare la evoluția gusturilor utilizatorului în funcție de gen și perioadă |
| **Scenarii Potrivite**                      | Schimbări frecvente în preferințele utilizatorilor     | Modelează pattern-uri secvențiale de vizionare a filmelor | Potrivit pentru utilizatorii cu preferințe specifice de gen și perioadă |
| **Feedback de la Utilizator**               | Angajament, rate-uri de clicuri, feedback de la utilizator | Angajament, rate-uri de clicuri, feedback de la utilizator | Feedback-ul utilizatorului este crucial pentru adaptarea recomandărilor |
