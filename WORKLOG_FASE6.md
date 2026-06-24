# WORKLOG — Fase 6: Fine-tuning end-to-end

Diario di lavoro per la stesura del progress report finale.
Ogni voce ha data, titolo e descrizione estesa del ragionamento.

---

## 2026-06-17 — Setup ambiente su nuova macchina e verifica struttura

### Contesto

Gli script di Fase 6 erano stati scritti e parzialmente testati su un altro computer (di Laura). Il lavoro riprende su una nuova macchina con GPU dedicata.

### Ambiente verificato

- Sistema operativo: Windows 10
- GPU: NVIDIA GTX 1060 6 GB VRAM (CUDA 12.6)
- Python 3.12.10 nel virtualenv `.venv`
- torch 2.6.0+cu124 già installato e funzionante (`torch.cuda.is_available()` → True)
- Tutte le librerie necessarie già presenti: timm 1.0.27, transformers 5.12.1, peft 0.19.1, h5py 3.16.0, scikit-learn 1.9.0, albumentations 2.0.8, pandas 3.0.3, matplotlib 3.11.0, seaborn 0.13.2, opencv-python 4.13.0, pillow 12.2.0, numpy 2.4.4, accelerate 1.14.0, safetensors 0.8.0.

### Struttura del repo verificata

La struttura corrisponde a quella progettata, con un'unica discrepanza minore: i tre script di test (`run_smoke_test.py`, `test_batchnorm_freeze.py`, `test_orchestration.py`) si trovano dentro `experiments/` anziché direttamente in `phase 6/`. Non è un problema funzionale.

Confermata la struttura asimmetrica dei crop:
- Falciparum e Vivax: sottocartelle pulite `G/R/S/T`
- Ovale: ha anche `G_R`, `R_T`, `T_R`, `diagnostics/`, `report.csv`
- Malariae: ha anche `R_T`, `S_T`, `diagnostics/`, `report.csv`, e **due file `.png` sciolti direttamente in `crops/`** (rilevante per il punto successivo)

---

## 2026-06-17 — Fix portabilità path nei CSV degli split

### Problema

I CSV degli split (generati su un altro computer) contengono path assoluti Windows-specifici nella colonna `filepath`, del tipo:

```
C:\Users\laura\OneDrive\...\MP-IDB-...\Falciparum\crops\R\1307210661-0001-R_0.png
```

Su questa macchina il progetto sta in `D:\DESKTOP\TIROCINIO\malaria-detection-pipeline\`, quindi quei path non esistono. Il training sarebbe fallito al primo tentativo di aprire un'immagine.

### Vincolo

Non si possono rigenerare i CSV: gli split sono patient-aware (costruiti garantendo che immagini dello stesso paziente non finiscano in train e validation insieme) e confrontabili con i risultati di Fase 5. Alterarli invaliderebbe il confronto tra pipeline.

### Soluzione implementata

Modifica a `scripts/phase 6/data/dataset.py`: aggiunta della funzione `_relocate_path` e applicazione su tutta la colonna `filepath` dentro `MalariaDataset.__init__`, prima di qualsiasi accesso al disco.

**Strategia di ancoraggio al nome della specie.** L'idea è trovare il primo componente del path che corrisponde a un nome di specie (`Falciparum`, `Malariae`, `Ovale`, `Vivax`) e ricostruire il path da quel punto in poi sotto `_BASE_ROOT`. Questo funziona perché:
- La struttura del path dopo la specie è sempre `[specie]/crops/[stage]/filename.png`
- I nomi delle specie non compaiono nei path precedenti alla directory della specie (né in `MP-IDB-...` né in `Users/laura/...`)
- Il confronto avviene su componenti interi del path (non su sottostringhe), eliminando falsi positivi

Alternativa scartata: sostituire la parte iniziale del path fino alla cartella radice del progetto. Non funziona perché le due macchine usano nomi di cartella radice diversi (`MP-IDB-...` vs `malaria-detection-pipeline`).

`_BASE_ROOT` viene derivato autonomamente in `dataset.py` senza importare `config.py`, usando `Path(__file__).resolve().parent.parent.parent.parent` (quattro livelli sopra `data/dataset.py` portano alla root del progetto). Questo evita dipendenze circolari e non richiede modifiche ai call site nei script di orchestrazione.

**Effetto collaterale positivo**: se i CSV venissero un giorno rigenerati su questa macchina (path già corretti), la funzione troverebbe comunque la specie nel path e ricostruirebbe lo stesso percorso — nessun errore.

### File modificato

`scripts/phase 6/data/dataset.py` — aggiunta funzione `_relocate_path` (righe 13-25) e chiamata in `MalariaDataset.__init__` (riga 55).

### Verifica

Testato manualmente su path reali estratti dai CSV: sia per Falciparum (`crops/R/...`) sia per Malariae (`crops/T/...`), il path ricostruito esiste su disco (`Path.exists()` → True).

---

## 2026-06-17 — Smoke test: prima verifica pipeline su questa macchina

### Configurazione

- Modello: ConvNeXt (torchvision, backbone convnext_tiny)
- Modalità: head_only (backbone completamente congelato, solo la testa lineare viene allenata)
- Fold: 1
- Epoche: 1 (test rapido, non per risultati definitivi)
- Batch size: 8
- Device: cuda (GTX 1060)

### Risultati

**White-box test (verifica componenti isolati):**

- Shape del batch: corretta `(8, 3, 224, 224)`
- Loss iniziale: 1.3919, con ln(4) = 1.3863. La vicinanza al logaritmo del numero di classi è un segnale sano: un modello appena inizializzato produce distribuzioni quasi uniformi sui 4 output, quindi la cross-entropy attesa è ln(4). Uno scarto troppo grande indicherebbe un'inizializzazione anomala.
- Gradiente sul backbone: assente (confermato il freezing)
- Gradiente sulla testa: presente e non nullo (la testa si aggiorna)
- Parametri trainable: 3,076 su 27,823,204 totali (0.01%) — coerente con il freezing del backbone

**Black-box test (pipeline completa):**

- Epoch 1: train_loss 1.0756, train_acc 0.6937, val_loss 0.5611, val_acc 0.9143
- Valutazione sul test held-out (58 campioni): Accuracy 0.9828, **F1 macro 0.9788**, MCC 0.9767

I risultati sono coerenti con quelli di Fase 5 (ConvNeXt + Logistic Regression aveva ottenuto F1 macro ≈ 0.979 intra-dataset), confermando che il backbone ConvNeXt ha rappresentazioni altamente discriminative già con una sola epoca di adattamento della testa.

Output salvati in: `results/tuning/smoke_test/ConvNeXt/head_only/fold1/` (metrics.json, history.json, classification_report.txt, confusion_matrix.png).

I pesi di ConvNeXt sono stati scaricati da torchvision e sono ora in cache locale (`C:\Users\vdell\.cache\torch\hub\checkpoints\convnext_tiny-983f1562.pth`) — i run successivi non richiederanno il download.

---

## 2026-06-17 — Analisi pianificazione esperimenti completi (decisioni metodologiche aperte)

### Volume totale di esperimenti

**Intra-dataset** (run_intra.py): 32 run totali
- ResNet50 e ConvNeXt: 2 modalità (head_only, full) × 2 fold = 4 run ciascuno
- Swin-T, ViT-B, RedDino, DinoBloom: 3 modalità (head_only, full, lora) × 2 fold = 6 run ciascuno
- (LoRA non disponibile per ResNet50 e ConvNeXt perché sono architetture CNN senza moduli attention lineari su cui applicare gli adattatori)

**OOD** (run_ood.py): 144 run totali
- 16 combinazioni modello×modalità × 9 coppie sorgente→target
- Le 9 coppie OOD testano la trasferibilità della rappresentazione appresa su una specie verso un'altra specie non vista in training

**Totale: 176 run.**

### Analisi VRAM e rischi per ogni combinazione

La VRAM durante il training è occupata da quattro componenti: pesi del modello, attivazioni intermedie (trattenute per il backward in modalità `full`), gradienti, e stato dell'ottimizzatore AdamW (due tensori aggiuntivi per ogni parametro trainable, pari a 2× la memoria dei pesi trainable).

Le architetture transformer presentano un collo di bottiglia specifico: le **attention map**, di dimensione `batch × heads × seq_len × seq_len`. In modalità `full` queste devono essere trattenute in memoria per tutti i layer simultaneamente per calcolare i gradienti. La dimensione cresce quadraticamente con `seq_len`, che dipende dalla risoluzione dell'immagine e dalla patch size.

**DinoBloom è il caso critico.** Con immagine 518px e patch_size=14, la sequence length è (518/14)² + 1 = 1370 token (vs 197 di ViT-B a 224px). Le attention map per un singolo layer a batch=32 occupano circa 2.9 GB — quasi metà dell'intera VRAM disponibile. Il full fine-tuning di DinoBloom a batch=32 è quasi certamente infeasibile su 6 GB; il solo input batch occupa già ~1 GB.

**ViT-B e RedDino** (224px, ~86M params) in modalità `full` con batch=32: stimato ~2.3 GB per le sole attention map, più i pesi e lo stato AdamW, porta a un totale che rischia di superare i 6 GB.

**Modelli CNN** (ResNet50, ConvNeXt, Swin-T) non presentano questo problema — le feature map convoluzionali crescono meno aggressivamente con la batch size.

**Sintesi rischio per batch size corrente (BATCH_SIZE=32 in config.py):**

| Modello | head_only | full | lora |
|---|---|---|---|
| ResNet50 | sicuro | sicuro | n/a |
| ConvNeXt | sicuro | sicuro | n/a |
| Swin-T | sicuro | ridurre a 16 | sicuro |
| ViT-B | sicuro | ridurre a 8 | sicuro |
| RedDino | sicuro | ridurre a 8 | sicuro |
| DinoBloom | ridurre a 8-16 | quasi certamente OOM | ridurre a 4-8 |

### Decisioni metodologiche da implementare (aperte al 2026-06-17)

Le seguenti modifiche sono state pianificate ma non ancora implementate:

1. **Riduzione BATCH_SIZE a 16** come default sicuro per i modelli 224px. Per recuperare l'effetto batch=32 senza consumare più VRAM si userà il **gradient accumulation**: si accumula il gradiente su 2 mini-batch da 16 prima di fare `optimizer.step()`, ottenendo matematicamente lo stesso aggiornamento di un batch da 32. Questa tecnica non cambia la convergenza del modello, cambia solo la frequenza degli aggiornamenti.

2. **Meccanismo di resume nel loop di run_intra.py**: se il processo si interrompe (per OOM o altri motivi), attualmente bisognerebbe ripartire dall'inizio. Aggiungere un controllo "skip se `metrics.json` già esiste" permette di riprendere da dove ci si era fermati.

3. **Test isolato di DinoBloom head_only** con batch=8 prima di includerlo nel loop completo — per verificare empiricamente se il forward pass a 518px è sostenibile sulla GPU disponibile.

---

## 2026-06-17 — Gradient accumulation, batch size per modello, resume mechanism

### Motivazione

Il `BATCH_SIZE = 32` definito globalmente in config.py sarebbe stato insostenibile per i modelli transformer in modalità `full` fine-tuning, dove tutte le attention map di tutti i layer devono essere trattenute in memoria simultaneamente per calcolare i gradienti durante il backward pass. Per ViT-B e RedDino (224px, 197 token) il rischio era già alto; per DinoBloom (518px, 1370 token) era quasi certo l'OOM.

### Soluzione: batch size per modello + gradient accumulation

Il **gradient accumulation** è una tecnica che permette di simulare un batch grande usando mini-batch più piccoli: si eseguono N forward+backward con mini-batch di dimensione B (dividendo la loss per N prima del `.backward()`, così i gradienti si sommano correttamente), e solo alla fine si chiama `optimizer.step()`. Il risultato matematico è identico a un batch di dimensione N×B.

I parametri `"batch_size"` e `"grad_accum_steps"` sono stati aggiunti a ciascun modello in `MODEL_CONFIG`, mantenendo un effective batch = 32 per tutti:

| Modello | batch_size | grad_accum_steps | Motivazione |
|---|---|---|---|
| ResNet50, ConvNeXt, Swin-T | 16 | 2 | CNN: nessun collo di bottiglia attention |
| ViT-B, RedDino | 8 | 4 | 197 token: attention map ~190 MB/layer a batch=32 |
| DinoBloom | 4 | 8 | 1370 token: input batch ~1 GB a batch=32 |

### File modificati

- `config.py`: aggiunti `"batch_size"` e `"grad_accum_steps"` a ogni entry di `MODEL_CONFIG`; `BATCH_SIZE` globale mantenuto solo come fallback con valore 16.
- `training/trainer.py`: `train_one_epoch` riscritta con logica di accumulation (`zero_grad` spostato fuori dal loop, loss scalata per `1/grad_accum_steps` prima di `.backward()`, `optimizer.step()` ogni N iterazioni o alla fine del loop); `train_model` aggiornata con parametro `grad_accum_steps=1`.
- `experiments/run_intra.py`: aggiunto skip-if-done (check `metrics.json` prima di ogni run), `output_dir` spostato all'inizio della funzione, batch_size e grad_accum_steps letti da `MODEL_CONFIG`.
- `experiments/run_ood.py`: stesse modifiche di run_intra.py.

### Dettaglio implementativo: scaling della loss

Il punto più sottile dell'accumulation è la divisione per `grad_accum_steps`. Se si accumulano i gradienti di N batch senza scalare, il gradiente risultante è N volte più grande del gradiente calcolato sul batch intero, e il learning rate effettivo aumenta di N. La divisione per N normalizza questa somma in una media, ottenendo lo stesso aggiornamento del batch originale. Le metriche (loss e accuracy) vengono invece tracciate sulla loss *non scalata*, in modo che i valori stampati a ogni epoca restino confrontabili.

---

## 2026-06-17 — Test DinoBloom head_only a 518px: verifica VRAM

### Configurazione

- Modello: DinoBloom (timm, `hf-hub:1aurent/vit_base_patch14_224.dinobloom`)
- Modalità: head_only
- Image size: 518px (patch_size=14 → 1370 token)
- Batch size: 4, grad_accum_steps: 8 (effective batch = 32)
- Epoche: 2 (test di fattibilità, non run definitivo)
- fold: 1 (train: 111 campioni, val: 35 campioni)

### Risultati

- VRAM dopo caricamento modello: **0.35 GB** (86M × 4 byte ≈ 344 MB, coerente)
- VRAM picco durante training: **0.58 GB** — molto al di sotto del limite di 6 GB
- Epoch 1: train_loss 1.43, train_acc 0.38, val_loss 0.66, val_acc 0.83
- Epoch 2: train_loss 0.38, train_acc 0.92, val_loss 0.50, val_acc 0.89
- Nessun OOM.

### Interpretazione

Il picco di soli 0.58 GB è spiegabile: in modalità `head_only` il backbone è congelato (`requires_grad=False` su tutti i parametri del backbone). PyTorch non costruisce il grafo computazionale per quelle operazioni, quindi le attivazioni intermedie vengono liberate layer per layer man mano che il forward avanza. L'unico grafo trattenuto in memoria è quello della piccola testa lineare (768→4 = 3K parametri). La dimensione dell'immagine (518px vs 224px) non influenza il consumo di VRAM in modalità head_only quanto influenzerebbe la modalità full.

### Caso ancora aperto: DinoBloom full fine-tuning

Con backbone non frozen, tutte le activation map di tutti i 12 layer devono essere trattenute simultaneamente per il backward. Con 1370 token e batch=4, le attention map (4 × 12 × 1370 × 1370 × 4 byte) teoricamente occupano già ~1.7 GB, a cui si aggiungono pesi, gradienti e stato dell'ottimizzatore (2× i pesi per AdamW). Il test empirico è ancora da fare prima di includere DinoBloom+full nel loop completo.

Output di test salvati in: `results/tuning/dinobloom_test/DinoBloom/head_only/fold1/`

---

## 2026-06-18 — Completamento loop intra-dataset (30 run)

### Esecuzione

Il loop `run_intra.py` ha eseguito tutte le 30 combinazioni pianificate (6 modelli × modalità disponibili × 2 fold informativi, escluso DinoBloom+full). Il processo si è interrotto dopo 29/30 run per timeout del background task, con `DinoBloom/lora/fold2` in stato parziale (best_model.pt salvato, metrics.json assente). Un secondo avvio ha completato il run mancante grazie al resume mechanism (skip se metrics.json esiste).

Output salvati in: `results/tuning/intra/<modello>/<modalità>/fold<N>/` — per ogni run: `metrics.json`, `history.json`, `classification_report.txt`, `confusion_matrix.png`, `best_model.pt`.

### Risultati completi

| Modello | Modalità | F1 fold1 | F1 fold2 | Media F1 |
|---|---|---|---|---|
| ConvNeXt | head_only | 0.9583 | 0.9788 | **0.9686** |
| ConvNeXt | full | 0.9188 | 0.9594 | 0.9391 |
| DinoBloom | lora | 0.9375 | 0.9602 | 0.9489 |
| ResNet50 | head_only | 0.9384 | 0.9583 | 0.9484 |
| DinoBloom | head_only | 0.9053 | 0.9788 | 0.9421 |
| Swin-T | full | 0.9384 | 0.9188 | 0.9286 |
| Swin-T | head_only | 0.9188 | 0.9188 | 0.9188 |
| Swin-T | lora | 0.9188 | 0.9188 | 0.9188 |
| ResNet50 | full | 0.9036 | 0.9249 | 0.9143 |
| RedDino | full | 0.8963 | 0.9384 | 0.9174 |
| RedDino | lora | 0.8392 | 0.8944 | 0.8668 |
| ViT-B | full | 0.8939 | 0.8750 | 0.8845 |
| ViT-B | lora | 0.7477 | 0.7001 | 0.7239 |
| ViT-B | head_only | 0.7276 | 0.6881 | 0.7079 |
| RedDino | head_only | 0.5687 | 0.8560 | 0.7124 |

DinoBloom+full è escluso perché non ancora testato empiricamente per VRAM.

### Osservazione 1: ConvNeXt head_only è il modello migliore

ConvNeXt in modalità head_only raggiunge F1 medio 0.9686, con un picco di 0.9788 su fold2 — identico al risultato del smoke test e coerente con i risultati di Fase 5 (ConvNeXt+Logistic Regression, F1 ≈ 0.979). La coerenza tra Fase 5 e Fase 6 è un segnale positivo: le feature estratte da ConvNeXt sono talmente discriminative che già una testa lineare statica (Fase 5) o pochissimi parametri allenati (Fase 6 head_only) raggiungono lo stesso livello di performance.

Il fatto che `full` sia peggio di `head_only` per ConvNeXt (e ResNet50) è spiegabile con l'overfitting: con meno di 120 campioni di training e 28M parametri tutti trainable, il modello impara a memoria il training set senza generalizzare meglio sul test. Il backbone pretrained su ImageNet fornisce già feature eccellenti che non guadagnano da aggiustamenti su dati così scarsi.

### Osservazione 2: Swin-T — 5 run con risultati identici

Cinque dei sei run di Swin-T producono esattamente gli stessi risultati (F1=0.9188, Accuracy=0.9310, MCC=0.9073), con gli stessi errori sugli stessi campioni (3 Ovale e 1 Malariae). Questo non è un bug. La spiegazione è la seguente:

Il test set ha 58 campioni fissi. Le feature del backbone Swin-T pretrained sono talmente discriminative che qualsiasi testa lineare ragionevole classifica correttamente gli stessi 54 campioni "facili" e sbaglia sugli stessi 4 campioni "ambigui". Le training curve di head_only e lora sono diverse (epoche diverse, loss diverse), ma convergono alla stessa soluzione sul test set perché quei 4 campioni sono strutturalmente difficili da separare con queste feature. Con un dataset di test così piccolo, bastano 4 campioni borderline per produrre metriche identiche. La conferma: `full fold1` riesce a classificarne correttamente uno in più (F1=0.9384), grazie all'aggiornamento del backbone che modifica abbastanza la rappresentazione di quel campione specifico.

### Osservazione 3: ViT-B — head_only peggio di full (fenomeno noto)

ViT-B in head_only e lora raggiunge solo F1 ~0.70-0.75, mentre in full arriva a ~0.875-0.894. Questo comportamento è l'opposto di quanto osservato nei CNN ed è una caratteristica nota dei transformer puri: il CLS token di ViT è ottimizzato per il task di pretraining (classificazione ImageNet-21k) e non è linearmente separabile per task nuovi senza adattare il backbone. I CNN costruiscono feature gerarchiche (texture, bordi, forme) che trasferiscono naturalmente; il ViT costruisce rappresentazioni globali tramite attention pattern dipendenti dal dominio, che richiedono fine-tuning end-to-end per essere riadattate. Questo implica che, per dataset piccoli dove l'overfitting è un rischio, il ViT è una scelta architetturale più problematica dei CNN.

### Osservazione 4: modelli domain-specific (RedDino, DinoBloom) non battono i generici

RedDino (specializzato per istopatologia) e DinoBloom (specializzato per microscopia di sangue) non superano ConvNeXt pretrained su ImageNet. Con meno di 120 campioni di training, il vantaggio del pretraining domain-specific non riesce a emergere: il dataset è troppo piccolo per differenziare i modelli sul loro adattamento al dominio. È possibile che con più dati (o con tecniche di augmentation più aggressive) i modelli specializzati sarebbero avvantaggiati.

### Caso ancora aperto

DinoBloom+full non è stato eseguito. Il test di VRAM per questa combinazione è ancora pendente.

---

## 2026-06-19 — Avvio loop OOD e aggiornamento run_ood.py

### Bug fix in evaluate.py (rilevato durante il loop OOD)

Prima di avviare il loop, è emerso un bug in `scripts/phase 6/evaluation/evaluate.py` che nel contesto OOD causava il crash di ogni run dopo 50 epoche di training completate correttamente.

**Causa**: `classification_report(all_labels, all_preds, target_names=SPECIES)` riceve 4 nomi di classe, ma nel contesto OOD il test set contiene solo immagini di una specie (la target) e il modello predice quasi esclusivamente la specie sorgente. Quindi sklearn vede al massimo 2 class ID unici in `all_labels + all_preds`, e rifiuta di mappare 4 nomi su 2 classi. Lo stesso problema si ripercuoteva su `f1_score` (che mediava su meno di 4 classi) e `confusion_matrix` (produceva una matrice 2×2 invece di 4×4).

**Fix**: aggiunto `labels=list(range(len(class_names)))` e `zero_division=0` a tutte e tre le chiamate. In questo modo sklearn usa sempre le 4 classi fisse; le classi assenti contribuiscono con F1=0, il che è il comportamento corretto. Il fix è backward-compatible con le run intra-dataset (dove tutte e 4 le classi sono sempre presenti nel test set).

File modificato: `scripts/phase 6/evaluation/evaluate.py`, righe 46-50.

---

### Decisione

Il loop intra-dataset è completato (30/30 run). Il passo successivo è il loop OOD. La scelta era tra (A) testare prima DinoBloom+full in isolamento per verificare la VRAM, oppure (B) avviare direttamente il loop OOD escludendo DinoBloom+full. È stata scelta l'opzione B: il test empirico della VRAM per DinoBloom+full è differito, e la combinazione viene esclusa dal loop OOD con lo stesso meccanismo già adottato in `run_intra.py`.

La motivazione è pragmatica: DinoBloom+full avrebbe aggiunto al massimo 9 run OOD (una per coppia), tutte con il rischio concreto di OOM. Le informazioni scientifiche ottenute sarebbero marginali rispetto al rischio di interrompere un loop da 135 run. Il valore comparativo primario viene già coperto da DinoBloom+head_only e DinoBloom+lora.

### Modifica a run_ood.py

File modificato: `scripts/phase 6/experiments/run_ood.py`

Aggiunte due modifiche al blocco `__main__`, per allinearlo con `run_intra.py`:

1. **Skip DinoBloom+full** — aggiunto lo stesso guard già presente in `run_intra.py`:
   ```python
   if model_name == "DinoBloom" and fine_tune_mode == "full":
       print(f"\nSKIP DinoBloom+full — da testare separatamente (VRAM)")
       continue
   ```

2. **Try/except attorno a ogni run** — senza di esso, un singolo errore (OOM, file corrotto, ecc.) interromperebbe l'intero loop da 135 run. Con il try/except, l'errore viene stampato con traceback completo e il loop prosegue alla run successiva.

### Conteggio run attive

- 6 modelli × 3 modalità = 18 combo
- −2: ResNet50+lora e ConvNeXt+lora (architetture CNN, non supportano LoRA)
- −1: DinoBloom+full (skip VRAM)
- = 15 combo attive × 9 coppie OOD = **135 run totali**

Le 9 coppie OOD sono: Falciparum→Vivax, Falciparum→Ovale, Falciparum→Malariae, Vivax→Falciparum, Vivax→Ovale, Vivax→Malariae, Ovale→Falciparum, Ovale→Vivax, Ovale→Malariae. Il resume mechanism già presente in `run_ood.py` permette di interrompere e riprendere il loop senza ripetere le run già completate.
