# Numele proiectului: [Numele proiectului]  
  
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
  
- Modelul 3  
  - Descrieți pe scurt fiecare model utilizat în cadrul proiectului. Includeți următoarele:  
    - Numele modelului  
    - Scopul/cazul de utilizare al modelului  
    - Principiul de funcționare de bază (opțional)  
  
  
## Differences  
| Caracteristică/Model         | TD Movie Recommender                         | Recomandare de Filme Bazată pe SARSA           |  
|------------------------------|---------------------------------------------|-----------------------------------------------|  
| **Abordare de Învățare**     | Învățare prin Diferență Temporală (TD)      | Învățare prin Stare-Actiune-Reward-Stare-Actiune (SARSA) |  
| **Avantaje**                 |                                             |                                               |  
| *Adaptabilitate*             | - Se adaptează la preferințele utilizatorului în timp. | - Se adaptează continuu la schimbările comportamentului utilizatorului. |  
| *Complexitatea Algoritmului* | - Relativ simplu și ușor de implementat.    | - Gestionează secvențe de acțiuni, ceea ce poate duce la recomandări mai nuanțate. |  
| *Precizia Predictivă*        | - Eficient în scenarii cu preferințe stabile. | - Potențial mai precisă, deoarece ia în considerare secvențe de acțiuni și recompense. |  
| *Explorare vs. Exploatare*    | - Menține un echilibru între explorarea filmelor noi și exploatarea preferințelor cunoscute. | - Menține un echilibru explicit între explorare și exploatare prin învățarea politicilor. |  
| **Dezavantaje**              |                                             |                                               |  
| *Sensibilitate la Datele Inițiale* | - Recomandările inițiale pot fi mai puțin precise până când se adună suficiente date. | - Este posibil să fie nevoie de mai multe date pentru a învăța eficient secvențe de acțiune. |  
| *Receptivitate la Schimbare* | - S-ar putea să se adapteze mai greu la schimbările bruște ale preferințelor utilizatorilor. | - Mai receptivă la schimbările imediate în comportamentul utilizatorului datorită învățării pe bază de politici. |  
| *Complexitate Computațională* | - Complexitate computațională mai mică.    | - Complexitate de calcul mai mare datorită luării în considerare a perechilor stare-acțiune. |  
| *Potrivire pentru Cazurile de Utilizare* | - Mai potrivită pentru mediile stabile în care preferințele utilizatorilor nu se schimbă rapid. | - Ideală pentru mediile dinamice în care preferințele utilizatorilor se schimbă frecvent. |  
  
  
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
  
## Model Details  
| Caracteristică                   | TD Movie Recommender                              | Recomandare de Filme Bazată pe SARSA                   |  
|---------------------------------|----------------------------------------------------|-------------------------------------------------------|  
| **Arhitectură și Design**       | - Bazată pe învățarea prin Diferență Temporală (TD). | - Folosește algoritmul SARSA (Stare-Actiune-Recompensă-Stare-Actiune). |  
|                                 | - Menține valorile Q pentru fiecare film.           | - Menține valorile Q pentru perechi stare-acțiune (secvențe de filme). |  
|                                 | - Actualizează valorile Q utilizând evaluările utilizatorilor ca recompense. | - Actualizează valorile Q pe baza tranzițiilor între perechile stare-acțiune. |  
| **Formatul și Cerințele Datelor de Intrare** | - Necesită fișierele `movies.dat` și `ratings.dat`. | - Necesită, de asemenea, fișierele `movies.dat` și `ratings.dat`. |  
|                                 | - Datele sunt în format tabular, codate în 'latin-1'.  | - Datele au un format tabular similar, codate în 'latin-1'. |  
|                                 | - Datele cheie: MovieID, Titlu, Genuri pentru filme; UserID, MovieID, Evaluare pentru evaluări. | - La fel ca în cazul TD Movie Recommender. |  
| **Formatul și Interpretarea Datelor de Ieșire**  | - Oferă o listă de titluri de filme ca recomandări. | - Oferă o listă de titluri de filme ca recomandări. |  
|                                 | - Recomandările se bazează pe cele mai mari valori Q. | - Recomandările sunt derivate din valorile Q ale perechilor stare-acțiune. |  
  
## Training and Evaluation  
| Proces de Antrenament și Evaluare             | Recomandare de Filme Bazată pe Diferență Temporală (TD) | Recomandare de Filme Bazată pe SARSA        |  
|----------------------------------------------|--------------------------------------------------------|-------------------------------------------|  
| **Pregătirea Datelor și Prelucrarea**         | - Datele sunt încărcate din fișierele `movies.dat` și  | - Datele sunt încărcate din fișierele `movies.dat` și |  
|                                              |   `ratings.dat`.                                      |   `ratings.dat`.                                    |  
|                                              | - Datele sunt filtrate pentru a selecta coloanele     | - Datele sunt filtrate pentru a selecta coloanele    |  
|                                              |   relevante (UserID, MovieID, Rating).                |   relevante (UserID, MovieID, Rating).               |  
|                                              | - Se utilizează codificarea 'latin-1' din cauza      | - Se utilizează codificarea 'latin-1' din cauza     |  
|                                              |   formatului datelor.                                 |   formatului datelor.                                |  
|                                              | - Evaluările utilizatorilor sunt folosite ca recompense | - Evaluările utilizatorilor sunt folosite ca recompense |  
|                                              |   pentru antrenament.                                 |   pentru antrenament.                                |  
|                                              |                                                      |                                                     |  
| **Procedura de Antrenament**                  | - Modelul utilizează învățarea prin Diferență Temporală | - Modelul utilizează algoritmul SARSA (Stare-Actiune-  |  
|                                              |   (TD).                                              |   Recompensă-Stare-Actiune) pentru învățare.           |  
|                                              | - Valorile Q pentru fiecare film sunt inițializate   | - Valorile Q pentru perechi stare-acțiune sunt inițializate |  
|                                              |   aleator.                                           |   aleator.                                            |  
|                                              | - Pentru istoricul evaluărilor fiecărui utilizator,   | - Pentru istoricul evaluărilor fiecărui utilizator,    |  
|                                              |   valorile Q sunt actualizate folosind algoritmul TD. |   valorile Q sunt actualizate pe baza algoritmului SARSA. |  
|                                              | - Parametri: Rată de Învățare (alpha), factor de     | - Parametri: Rată de Învățare (alpha), factor de        |  
|                                              |   discount (gamma).                                  |   discount (gamma).                                   |  
|                                              |                                                      |                                                     |  
| **Metrici și Metode de Evaluare**            | - Evaluarea este adesea implicită, concentrată pe    | - Evaluarea este adesea implicită, cu accent pe capacitatea |  
|                                              |   capacitatea de a se adapta la preferințele         |   modelului de a se adapta la preferințele dinamice ale   |  
|                                              |   utilizatorului în timp.                           |   utilizatorului.                                     |  
|                                              | - Recomandările sunt generate pe baza valorilor Q   | - Recomandările sunt generate pe baza valorilor Q ale    |  
|                                              |   învățate.                                          |   perechilor stare-acțiune.                            |  
|                                              | - Nu există metrici de evaluare explicite; feedbackul | - Nu există metrici de evaluare explicite; feedbackul  |  
|                                              |   utilizatorului poate ghida rafinarea modelului.    |   utilizatorului poate ghida rafinarea modelului.     |  
|                                              | - Satisfacția utilizatorului și interacțiunea sunt  | - Satisfacția utilizatorului și interacțiunea sunt    |  
|                                              |   indicatori cheie ai succesului.                   |   indicatori cheie ai succesului.                    |  
  
## Results and Discussion  
| Aspect                                      | Recomandare de Filme Bazată pe Diferență Temporală (TD) | Recomandare de Filme Bazată pe SARSA        |  
|---------------------------------------------|--------------------------------------------------------|-------------------------------------------|  
| **Metrici de Performanță**                   | Implicit, satisfacția utilizatorului și interacțiunea  | Implicit, satisfacția utilizatorului și interacțiunea |  
| **Focus**                                   | Recomandări individuale pentru filme              | Recomandări sub forma secvențelor de filme   |  
| **Viteză de Adaptare**                      | Adaptare rapidă la preferințele care se schimbă rapid | Eficient în modelarea preferințelor dinamice |  
| **Scenarii Potrivite**                      | Schimbări frecvente în preferințele utilizatorilor | Modelează pattern-uri secvențiale de vizionare a filmelor |  
| **Feedback de la Utilizator**               | Angajament, rate-uri de clicuri, feedback de la utilizator | Angajament, rate-uri de clicuri, feedback de la utilizator |
