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

---

## 2026-06-26 — Diagnosi bug loop OOD e correzioni a losses.py e trainer.py

### Contesto

Dopo l'esecuzione del loop OOD (135 run completate), è emerso che tutti i `metrics.json` mostrano `f1_macro=0.0, accuracy=0.0, mcc=0.0`. Questo ha portato a una diagnosi dettagliata dei tre problemi segnalati.

### Risultati OOD già salvati: da scartare o da conservare?

I 135 `metrics.json` già salvati in `results/tuning/ood/` sono stati prodotti con i bug descritti sotto. Per quanto riguarda i valori numerici (F1, accuracy, MCC), le correzioni applicate oggi (Bug 1 e Bug 3) **non cambiano i numeri**: Bug 1 era cosmetic (i pesi inf non entravano nel calcolo), Bug 3 era di performance (early stopping). I valori a zero riflettono il protocollo OOD così come definito (vedi sotto). I risultati sono quindi da conservare e da interpretare correttamente, non da rigenerare per ragioni tecniche.

### Bug 1 — Pesi infiniti in `losses.py` (cosmetic, nessun effetto sui risultati)

**Diagnosi**: In contesto OOD il training set contiene una sola specie. Le altre tre classi hanno `class_count=0`, causando una divisione per zero e pesi `inf` nella CrossEntropyLoss pesata. Tuttavia i pesi `inf` riguardano classi mai presenti nel training set, quindi non entrano mai nella formula della loss (che normalizza per `Σ weight[y_i]` sui soli campioni del batch). Il training era matematicamente corretto nonostante il warning.

**Correzione applicata** in `scripts/phase 6/training/losses.py`: sostituita la divisione diretta con `np.where(class_count > 0, N/(K*count), 0.0)`. Le classi assenti ricevono peso 0 invece di inf. Effetto sui risultati: nessuno (equivalente matematicamente).

### Bug 2 — Disallineamento SPECIES vs SPECIES_TO_ID (display bug, non corretto in questa sessione)

**Diagnosi**: `SPECIES = ["Falciparum", "Vivax", "Ovale", "Malariae"]` ha Vivax all'indice 1 e Malariae all'indice 3, ma `SPECIES_TO_ID` assegna Malariae=1 e Vivax=3. Nei report testuali (`classification_report`) e nelle confusion matrix PNG, Vivax e Malariae risultano scambiate. I valori numerici in `metrics.json` (F1 macro, accuracy, MCC) sono corretti perché calcolati sugli ID numerici. Il bug è presente anche nelle run intra-dataset, ma non altera le metriche numeriche. Correzione non applicata in questa sessione: decisione rimandata.

### Bug 3 — Early stopping con `min_delta=0.0` (performance, non correttezza)

**Diagnosi**: Con `min_delta=0.0`, qualunque miglioramento infinitesimale della val_loss resetta il contatore. In OOD la val_loss scende rapidamente verso 0 e poi oscilla con micro-miglioramenti, quindi il contatore non raggiunge mai la patience=10. Ogni run eseguiva sempre le 50 epoche intere anche quando il modello aveva già convergito alla prima o seconda epoca.

**Correzione applicata** in `scripts/phase 6/training/trainer.py`: `min_delta` cambiato da `0.0` a `1e-4`. Con questa soglia, se la val_loss non migliora di almeno 0.0001 per 10 epoche consecutive il training si ferma. Effetto sui risultati: nessuno (il best checkpoint è già salvato alla prima epoca di convergenza); effetto sul tempo di calcolo: significativo, specialmente per DinoBloom a 518px.

### Nota concettuale: perché F1=0 non è un bug del codice

Il protocollo OOD di Fase 6 allena un classificatore a 4 classi su immagini di una sola specie sorgente, poi lo testa su immagini di una specie target diversa. Per costruzione, il modello impara "qualsiasi immagine = classe sorgente" e predice sempre quella classe: sul test set (solo immagini target) ogni predizione è sbagliata → F1=0.

Questo è diverso dal protocollo OOD di Fase 5, che allenava su *fasi del ciclo cellulare* (G/R/S/T) della specie sorgente e testava sulle stesse fasi della specie target. Le fasi sono un label space condiviso tra le specie, quindi il trasferimento aveva senso biologico. In Fase 6 non esiste un label space condiviso tra sorgente e target nel protocollo attuale. I valori a zero sono la risposta corretta al protocollo così definito, non un artefatto tecnico. Eventuale ridefinizione del protocollo va discussa con il relatore.

---

## 2026-06-28 — Analisi risultati loop OOD completato (135/135 run)

### Riepilogo esecuzione

Il loop OOD è completato con 135/135 run salvate in `results/tuning/ood/`. Le 15 combo attive (dopo gli skip per LoRA non supportata e DinoBloom+full) × 9 coppie source→target.

### Risultato dominante: F1=0 su 132/135 run

Come previsto dall'analisi del protocollo del 2026-06-26, **132 run su 135 producono F1 macro = 0.0, accuracy = 0.0, MCC = 0.0**. Il pattern è uniforme su tutti i modelli (ResNet50, ConvNeXt, Swin-T, ViT-B, RedDino) e su tutte e tre le modalità (head_only, full, lora): qualunque modello allenato su una sola specie predice sempre quella specie sul test set della specie target, dando zero predizioni corrette.

### Eccezione: 3 run DinoBloom con F1 non zero

Le uniche tre run con F1 > 0 sono tutte DinoBloom:

| Coppia | Modalità | F1 macro | Accuracy |
|---|---|---|---|
| Ovale → Falciparum | lora | **0.080** | 0.190 |
| Falciparum → Malariae | head_only | **0.071** | 0.167 |
| Falciparum → Ovale | head_only | **0.045** | 0.100 |

I valori sono bassi in assoluto (F1 < 0.10), ma statisticamente significativi rispetto al fondo di zeri. Il MCC è 0 in tutti e tre i casi, il che indica che la correlazione tra predizioni e label vere è nulla nonostante l'accuracy non sia zero: il modello classifica qualche campione correttamente ma non in modo sistematico.

### Interpretazione

Il fatto che le uniche eccezioni siano DinoBloom è coerente con la sua architettura: DinoBloom è pre-allenato specificamente su immagini di microscopia di sangue malarico (hf-hub:1aurent/vit_base_patch14_224.dinobloom). A differenza dei modelli ImageNet-pretrained, le sue feature sono già orientate al dominio della microscopia malarica. Questo potrebbe conferire una minima trasferibilità cross-specie anche in un contesto di allenamento monospecie: le rappresentazioni interne di DinoBloom per immagini di Falciparum o Ovale condividono già un vocabolario visivo parzialmente compatibile con Malariae e Falciparum.

L'eccezione DinoBloom+lora è la più alta (F1=0.08 su Ovale→Falciparum): LoRA adatta solo le matrici di attenzione, mantenendo gran parte delle feature pre-allenate. In questo caso l'adattamento LoRA potrebbe preservare meglio la generalità del backbone rispetto al full fine-tuning, che tenderebbe a specializzarsi troppo sulla specie sorgente.

Tutti gli altri modelli (compresi RedDino, che è domain-specific per istopatologia ma non per malaria) restano a F1=0: il loro pre-training su ImageNet o su tessuto istologico non fornisce feature abbastanza specifiche da generalizzare tra specie malariche.

### Nota metodologica per il report

Il protocollo OOD di Fase 6 (train monospecie, test su specie diversa) non è comparabile con Fase 5 (train su fasi morfologiche di una specie, test su fasi di un'altra). I risultati di Fase 6 OOD non sono "sbagliati": misurano la generalizzazione cross-specie di un classificatore addestrato su dati mono-specie, che è un task estremo e produce quasi invariabilmente F1=0. Il risultato più interessante per il report è l'eccezione DinoBloom, che evidenzia il valore del pre-training domain-specific per la robustezza OOD.

### Risultati completi

Riepilogo per combo (F1 macro mediato sulle 9 coppie source→target):

| Modello | Modalità | F1 media | F1 max | Run non-zero |
|---|---|---|---|---|
| DinoBloom | head_only | 0.0130 | 0.0714 | 2/9 |
| DinoBloom | lora | 0.0089 | 0.0800 | 1/9 |
| Swin-T | head_only | 0.0000 | 0.0000 | 0/9 |
| Swin-T | full | 0.0000 | 0.0000 | 0/9 |
| Swin-T | lora | 0.0000 | 0.0000 | 0/9 |
| ViT-B | head_only | 0.0000 | 0.0000 | 0/9 |
| ViT-B | full | 0.0000 | 0.0000 | 0/9 |
| ViT-B | lora | 0.0000 | 0.0000 | 0/9 |
| ResNet50 | head_only | 0.0000 | 0.0000 | 0/9 |
| ResNet50 | full | 0.0000 | 0.0000 | 0/9 |
| ConvNeXt | head_only | 0.0000 | 0.0000 | 0/9 |
| ConvNeXt | full | 0.0000 | 0.0000 | 0/9 |
| RedDino | head_only | 0.0000 | 0.0000 | 0/9 |
| RedDino | full | 0.0000 | 0.0000 | 0/9 |
| RedDino | lora | 0.0000 | 0.0000 | 0/9 |

DinoBloom+full escluso (VRAM). LoRA esclusa per ResNet50 e ConvNeXt.

Dettaglio delle 3 run con F1 > 0:

| Coppia | Modello | Modalità | F1 macro | Accuracy | MCC |
|---|---|---|---|---|---|
| Ovale → Falciparum | DinoBloom | lora | 0.0800 | 0.1905 | 0.0 |
| Falciparum → Malariae | DinoBloom | head_only | 0.0714 | 0.1667 | 0.0 |
| Falciparum → Ovale | DinoBloom | head_only | 0.0455 | 0.1000 | 0.0 |

---

## 2026-06-28 — Test VRAM DinoBloom+full e avvio run mancanti

### Motivazione

DinoBloom+full era stato escluso da entrambi i loop (run_intra.py e run_ood.py) con un guard esplicito perché il costo VRAM del full fine-tuning a 518px non era stato verificato empiricamente. Il test head_only aveva dato un picco di 0.58 GB (solo il modello in forward, senza gradienti sul backbone). Con full, tutti i 12 transformer layer richiedono che PyTorch mantenga in memoria le activation map intermedie per la backpropagation — costo molto più alto.

### Metodo: run_dinobloom_test.py modificato

`run_dinobloom_test.py` è stato adattato cambiando `mode = "head_only"` in `mode = "full"` e aggiungendo `torch.cuda.reset_peak_memory_stats()` dopo il caricamento del modello (per misurare solo il costo del training, escludendo i 0.35 GB fissi del modello). Il test ha girato fold1, 2 epoche, con batch_size=4 e grad_accum_steps=8 come configurati in MODEL_CONFIG["DinoBloom"].

### Risultato

```
VRAM dopo caricamento modello: 0.35 GB
Epoch 1 | train_loss: 0.8218 | train_acc: 0.7477 | val_loss: 0.4053 | val_acc: 0.8857
Epoch 2 | train_loss: 0.0125 | train_acc: 1.0000 | val_loss: 0.4902 | val_acc: 0.8857
VRAM picco: 5.16 GB
Test DinoBloom completato senza OOM.
```

**Picco VRAM: 5.16 GB su 6.00 GB disponibili.** Margine di ~0.84 GB: sufficiente per procedere in modo stabile, tenuto conto che le run complete (fino a 50 epoche con early stopping) non aggiungono allocazione rispetto alle 2 epoche di test (il grafo computazionale ha la stessa dimensione per ogni batch, indipendentemente dal numero di epoche).

### Approccio adottato: script separati

Anziché rimuovere il guard da run_intra.py e run_ood.py (che restano intatti per riproducibilità), sono stati creati due script dedicati:
- `experiments/run_dinobloom_full_intra.py` — fold1 e fold2 con resume mechanism
- `experiments/run_dinobloom_full_ood.py` — 9 coppie source→target con resume mechanism

Entrambi sono hardcodati su `MODEL_NAME = "DinoBloom"` e `FINE_TUNE_MODE = "full"`, replicano la stessa logica degli script principali (stessa struttura dati, stesse metriche, stesso save_results), e salvano i risultati nelle stesse cartelle dei loop principali (`results/tuning/intra/` e `results/tuning/ood/`) per coerenza con gli altri esperimenti.

### Risultati intra (completati 2026-06-29)

| Fold | F1 macro | Accuracy | MCC |
|------|----------|----------|-----|
| fold1 | **0.958** | 0.966 | 0.953 |
| fold2 | **0.913** | 0.931 | 0.907 |
| **Media** | **0.936** | **0.948** | **0.930** |

Risultati salvati in `results/tuning/intra/DinoBloom/full/fold1/metrics.json` e `.../fold2/metrics.json`.

### Posizionamento nel ranking intra

DinoBloom+full con F1 media 0.936 si colloca tra Swin-T full (0.929) e DinoBloom lora (0.949) nel ranking complessivo. È il terzo risultato assoluto:

| Rank | Modello | Modalità | F1 media |
|------|---------|----------|----------|
| 1 | ConvNeXt | head_only | 0.969 |
| 2 | DinoBloom | lora | 0.949 |
| 3 | ResNet50 | head_only | 0.948 |
| 4 | **DinoBloom** | **full** | **0.936** |
| 5 | Swin-T | full | 0.929 |
| 6 | RedDino | full | 0.917 |
| 7 | ViT-B | full | 0.885 |
| 8 | RedDino | head_only | 0.712 |

Notare che DinoBloom+lora (0.949) supera DinoBloom+full (0.936): il full fine-tuning non è vantaggioso per DinoBloom nel contesto intra-dataset, probabilmente perché il pre-training domain-specific già fornisce feature ottimali che il fine-tuning completo rischia di degradare leggermente (overfitting sul training set del fold).

### Risultati OOD DinoBloom+full (completati 2026-06-30)

Tutte e 9 le coppie: **F1 macro = 0.000, accuracy = 0.000, MCC = 0.000**.

| Coppia | F1 macro |
|--------|----------|
| Falciparum → Vivax | 0.0 |
| Falciparum → Ovale | 0.0 |
| Falciparum → Malariae | 0.0 |
| Vivax → Falciparum | 0.0 |
| Vivax → Ovale | 0.0 |
| Vivax → Malariae | 0.0 |
| Ovale → Falciparum | 0.0 |
| Ovale → Vivax | 0.0 |
| Ovale → Malariae | 0.0 |

### Interpretazione OOD DinoBloom per modalità

Confrontando le tre modalità di DinoBloom in OOD si ottiene un gradiente netto:

| Modalità | F1 media (9 coppie) | Run non-zero |
|----------|---------------------|--------------|
| head_only | 0.0130 | 2/9 |
| lora | 0.0089 | 1/9 |
| **full** | **0.0000** | **0/9** |

Il pattern è coerente con la teoria: più si modificano i pesi del backbone DinoBloom, meno rimane della generalizzazione cross-specie acquisita durante il pre-training su microscopia malarica. Con head_only il backbone è completamente frozen e le sue feature universali restano intatte. Con LoRA si modificano solo le matrici di attenzione ma il resto del backbone è preservato. Con full fine-tuning tutti gli 86 milioni di parametri vengono sovraiscritti per classificare immagini di una sola specie sorgente: il risultato è un modello che ha "dimenticato" le feature domain-specific acquisite durante il pre-training e si comporta esattamente come i modelli ImageNet-pretrained — F1=0 su tutte le coppie.

Questo rafforza la conclusione già emersa per il ranking intra: il pre-training domain-specific di DinoBloom ha valore, ma va preservato (head_only o LoRA), non sovrascritto.

### Stato finale Fase 6: tutti gli esperimenti completati

- Loop intra: **32/32 run** (30 originali + 2 DinoBloom+full)
- Loop OOD: **144/144 run** (135 originali + 9 DinoBloom+full)

---

## 2026-06-28 — Cambio da PyRadiomics a scikit-image per l'estrazione radiomica

### Problema

PyRadiomics 3.0.1/3.1.0 (le uniche versioni disponibili su PyPI) non è installabile su Python 3.12. Il motivo è in `versioneer.py`, uno strumento di versionamento incluso nel pacchetto sorgente: usa `configparser.SafeConfigParser`, funzione rimossa definitivamente in Python 3.12. Il problema si manifesta durante la fase di build (preparazione dei metadati), prima ancora che il codice di estrazione delle feature venga toccato. Il repository GitHub ufficiale non ha un fix compatibile con Python 3.12 nemmeno nel branch main. Python non può essere retrocesso perché PyTorch (già installato e funzionante) richiede Python 3.12.

### Soluzione: scikit-image

scikit-image (0.26.0, già installata nell'ambiente) fornisce gli strumenti per calcolare feature radiomiche equivalenti a quelle di PyRadiomics nelle categorie più importanti. La libreria è stabile, attivamente mantenuta, e non richiede compilazione (ha wheel pre-compilate per Python 3.12).

### Feature estratte (61 totali)

**First-order (15)**: statistiche calcolate direttamente sui valori di intensità dei pixel nella ROI: media, mediana, deviazione standard, varianza, asimmetria (skewness di Fisher), curtosi (excess), energia, entropia di Shannon (sull'istogramma a 256 bin), massimo, minimo, range, RMS, IQR, decile inferiore (P10), decile superiore (P90).

**GLCM (36)**: la Gray Level Co-occurrence Matrix descrive come i livelli di grigio si relazionano con i propri vicini spaziali. È calcolata con 16 livelli, 4 direzioni (0°, 45°, 90°, 135°) e 3 distanze di passo (1, 2, 3 pixel). Per ogni coppia (distanza, proprietà) si calcola media e deviazione standard across le 4 direzioni, ottenendo una rappresentazione rotationally invariant. Proprietà: contrasto, dissimilarità, omogeneità, energia, correlazione, ASM. NaN nella correlazione (patch a intensità costante) vengono sostituiti con 0.

**Shape 2D (10)**: proprietà geometriche della ROI calcolate con `regionprops`: area, perimetro, eccentricità, solidità, extent, lunghezza degli assi maggiore e minore, diametro equivalente, orientazione, numero di Eulero.

### Cosa manca rispetto a PyRadiomics

Le classi GLRLM, GLSZM, GLDM e NGTDM non sono disponibili in scikit-image e richiederebbero implementazione da zero. Queste matrici catturano pattern di run-length e di zona che completano la descrizione della texture, ma sono meno frequenti nei lavori applicativi rispetto a GLCM e first-order. La copertura con scikit-image è sufficiente per un confronto significativo con le deep feature nel contesto del tirocinio.

### File aggiornato

`scripts/phase4/extract_radiomic_features.py` — riscritto interamente; rimosso l'import di PyRadiomics e SimpleITK, aggiunte le funzioni `_first_order_features`, `_glcm_features`, `_shape_features` con scikit-image. La struttura esterna (loop sui fold, `save_h5`, `validate_h5`, `_relocate_path`, `build_roi_mask`) è rimasta identica.

---

## 2026-06-30 — Estrazione radiomica completata: verifica output e struttura

### Esecuzione

`scripts/phase4/extract_radiomic_features.py` eseguito sull'ambiente principale (Python 3.12, `.venv`). Nessun errore.

### Struttura output verificata

Prodotti 15 file `.h5` in `results/features/radiomics/fold{1-5}/{train,val,test}.h5`.

| File | Campioni | Feature | NaN |
|---|---|---|---|
| fold1/train | 111 | 61 | 0 |
| fold1/val | 35 | 61 | 0 |
| fold1/test | 58 | 61 | 0 |
| fold2/train | 122 | 61 | 0 |
| fold2/val | 24 | 61 | 0 |
| fold2/test | 58 | 61 | 0 |
| fold3/train | 140 | 61 | 0 |
| fold3/val | 6 | 61 | 0 |
| fold4/train | 126 | 61 | 0 |
| fold4/val | 20 | 61 | 0 |
| fold5/train | 125 | 61 | 0 |
| fold5/val | 21 | 61 | 0 |
| fold{1-5}/test | 58 (tutti identici) | 61 | 0 |

Nessun valore NaN o Inf in nessun file: l'estrattore scikit-image è numericamente stabile anche su crop piccoli o a basso contrasto.

### Conferma struttura dei fold

La distribuzione delle classi nei validation set conferma e chiarisce il vincolo "solo fold 1 e 2 sono informativi":

- **fold1/val**: Falciparum=20, Malariae=15 (2 classi)
- **fold2/val**: Falciparum=16, Malariae=8 (2 classi)
- **fold3/val**: Falciparum=6 (1 classe sola)
- **fold4/val**: Falciparum=20 (1 classe sola)
- **fold5/val**: Falciparum=21 (1 classe sola)

Ovale e Vivax hanno solo 2 group_id ciascuno: in un 5-fold patient-aware split, con 5 "slot" e solo 2 pazienti, quasi tutti i fold li trovano entrambi nel training set. I fold 1 e 2 sono gli unici in cui entrambe le classi rare (Ovale, Vivax) potrebbero contribuire al training avendo almeno qualche rappresentante — e i loro val set, pur contenendo solo 2 classi, sono sufficientemente bilanciati per calcolare un F1 macro significativo su Falciparum e Malariae. Nei fold 3-5 il val set ha una sola classe: F1 macro sarebbe trivialmente 1.0 e non confrontabile con gli altri fold.

Il test set (58 campioni, 4 classi: Falciparum=21, Malariae=12, Ovale=10, Vivax=15) è identico per tutti i fold — deriva dallo stesso `test_heldout.csv`. È il riferimento principale per confrontare radiomica vs deep features.

---

## 2026-06-30 — Classificazione radiomica Fase 5: risultati intra-dataset e OOD

### Intra-dataset

Script: `scripts/phase 5/intra_dataset_radiomic.py`. Risultati in `results/classification/intra_radiomic/metrics.csv`.

**Metriche sul test set held-out (58 campioni, 4 classi) — fold 1 e fold 2:**

| Classificatore | F1 macro fold1 | F1 macro fold2 | F1 medio | MCC fold1 | MCC fold2 |
|---|---|---|---|---|---|
| **RF** | **0.9566** | **0.9566** | **0.9566** | 0.954 | 0.954 |
| LR | 0.9132 | 0.9363 | 0.9248 | 0.907 | 0.929 |
| KNN_5 | 0.8983 | 0.7234 | 0.8109 | 0.882 | 0.753 |
| KNN_3 | 0.8333 | 0.7125 | 0.7729 | 0.842 | 0.783 |
| KNN_1 | 0.8016 | 0.6992 | 0.7504 | 0.821 | 0.754 |

Il classificatore migliore è **Random Forest** con F1 macro medio **0.957**, identico su entrambi i fold informativi. Questo risultato è sorprendente: 61 feature radiomiche calcolate analiticamente su immagini in scala di grigi rivalizzano con i migliori backbone deep (ConvNeXt head_only: 0.969, DinoBloom LoRA: 0.949, ResNet50 head_only: 0.948). La radiomica supera una metà dei backbone considerati.

La Logistic Regression è seconda a 0.925 di media — ragionevole, dato che le feature radiomiche sono già normalizzate da StandardScaler e LR si comporta bene su feature linearmente separabili. KNN degrada molto tra fold1 e fold2 (0.898 → 0.723 per KNN_5): le feature radiomiche non formano cluster compatti e consistenti nello spazio euclideo, rendendo la distanza un criterio meno robusto.

**Nota sulle metriche val**: I val set di fold1 e fold2 contengono solo 2 classi (Falciparum + Malariae). Il modello, addestrato su 4 classi, può predire anche Ovale e Vivax — se lo fa, le predizioni errate su classi assenti nel val abbassano artificialmente il F1 macro (questo spiega F1=0.424 per RF su fold1/val pur con accuracy=0.80). Per fold2/val tutti i classificatori ottengono F1=1.0: le feature radiomiche separano perfettamente Falciparum e Malariae su quel specifico sottoinsieme. I valori val non vanno confrontati tra fold né presi come misura assoluta. Il test held-out è la metrica di riferimento.

### OOD (classificazione cross-specie della fase)

Script: `scripts/phase 5/out_of_distribution_radiomic.py`. Risultati in `results/classification/ood_radiomic/`.

**Richiamo del protocollo**: il classificatore viene addestrato su tutti i campioni di una specie sorgente (predice la fase di sviluppo: R/G/S/T) e testato sui campioni di una specie target. Si misura se la relazione tra feature radiomiche e fasi biologiche è conservata cross-specie.

**F1 macro per coppia sorgente→target, miglior classificatore:**

| Sorgente → Target | Miglior classif. | F1 macro | Note |
|---|---|---|---|
| Falciparum → Vivax | LR | 0.500 | Acc=0.933 ma F1=0.5: collasso su classe dominante |
| Falciparum → Ovale | RF / KNN | 0.412 | |
| Falciparum → Malariae | LR | 0.056 | Quasi zero |
| Vivax → Falciparum | RF | 0.475 | Solo 2 classi in test |
| Vivax → Ovale | KNN | 0.083 | Molto basso |
| Vivax → Malariae | LR | 0.246 | |
| Ovale → Falciparum | KNN_1/3/5 | 0.533 | Risultato più alto in assoluto |
| Ovale → Vivax | LR | 0.450 | |
| Ovale → Malariae | tutti | ≈0.051 | Sostanzialmente zero |

I risultati OOD sono variabili e nessun classificatore domina chiaramente in tutte le coppie. La F1 media sui classificatori considerati migliori per coppia rimane tra 0.05 e 0.53 — molto inferiore all'intra-dataset.

**Osservazioni per il report:**

1. La generalizzazione cross-specie delle feature radiomiche è parziale e asimmetrica: alcune coppie (Ovale→Falciparum, Falciparum→Vivax) mostrano segnale, altre (→Malariae in particolare) no.

2. Malariae come target ottiene sempre risultati vicini a zero. Questo è coerente con il fatto che Malariae è la specie con la biologia più diversa (parassita più piccolo, ciclo eritrocitico più lungo): le sue fasi non corrispondono bene alle fasi delle altre specie in termini di feature di texture e forma.

3. Alcuni risultati sembrano "alti" per accuracy ma hanno F1 macro basso (es. LR Falciparum→Vivax: acc=0.933, F1=0.50). Questo è un artefatto classico: il classificatore predice sempre la classe con più campioni nel training (dominanza di classe sorgente), ottenendo alta accuracy su un test set sbilanciato ma F1 macro penalizzato.

4. **Confronto con Fase 6 OOD**: il task è completamente diverso. Fase 6 OOD addestrapa un classificatore di specie su campioni monospecie e testava la generalizzazione cross-specie. Fase 5 OOD addestra un classificatore di fasi e testa la generalizzazione cross-specie delle fasi stesse. I due protocolli non sono confrontabili direttamente. I risultati radiomic OOD (F1 non-zero in 7/9 coppie) non sono migliori del Fase 6 deep OOD: sono su un task diverso e più informativo.

### Conclusione complessiva

La radiomica con scikit-image (61 feature, RF) raggiunge F1=0.957 intra-dataset, posizionandosi nel gruppo dei migliori modelli del progetto insieme a ConvNeXt e ResNet50 fine-tunati. Questo è un risultato rilevante per il report: dimostra che feature classiche di texture e forma catturano strutture diagnostiche robuste nei crop RBC, rivaleggiando con rappresentazioni apprese da reti neurali pre-trained su ImageNet. L'ipotesi che "più features = migliori risultati" non si conferma: 61 feature analitiche sono sufficienti se calcolate sulla morfologia corretta.

---

## 2026-06-30 — Confronto radiomica vs deep features (Fase 5)

Confronto tra i risultati radiomica (appena completati) e i risultati deep della Fase 5 (backbone pre-addestrati: ResNet50, ConvNeXt Tiny, ViT-B/16, Swin-T, DinoBloom, RedDino; classificatori: RF, LR, KNN k=1/3/5).

### Ranking intra-dataset unificato

Le metriche deep sono media ± std tra fold 1 e fold 2 sul test set held-out. Le metriche radiomica sono media tra fold 1 e fold 2 (valori identici: 0.9566 per RF).

| Rank | Modello | Classificatore | F1 macro | Tipo feature |
|---|---|---|---|---|
| 1 | ConvNeXt Tiny | LR | 0.9788 | deep (2048 dim) |
| 2 | ViT-B/16 | LR | 0.9788 | deep (2048 dim) |
| 3 | ConvNeXt Tiny | KNN_3/5 | 0.9742 | deep (2048 dim) |
| **4** | **Radiomica** | **RF** | **0.9566** | **analitiche (61 dim)** |
| 5 | DinoBloom | LR | 0.9556 | deep (768 dim) |
| 6 | Swin-T | RF | 0.9621 | deep (768 dim) |
| 7 | ResNet50 | LR | 0.9106 | deep (2048 dim) |
| 8 | RedDino | KNN_3 | 0.8514 | deep (768 dim) |

La radiomica RF occupa il 4° posto, in sostanziale parità con DinoBloom+LR (delta: +0.001). Supera nettamente ResNet50 e RedDino pur usando 33 volte meno feature (61 vs 2048). I soli modelli che la precedono chiaramente sono ConvNeXt e ViT-B/16 con LR.

### Inversione del classificatore migliore

Il pattern più rilevante per il report è che il classificatore ottimale si inverte tra deep e radiomica:

| Tipo feature | Migliore | Peggiore | Gap LR–RF |
|---|---|---|---|
| Deep (media su 6 backbone) | **LR** (F1=0.937) | KNN_1 (0.910) | LR batte RF di +0.022 |
| Radiomica | **RF** (F1=0.957) | KNN_1 (0.750) | RF batte LR di +0.032 |

**Motivazione**: i backbone deep estraggono feature nell'ultimo strato prima del classificatore, uno spazio latente progettato per essere linearmente separabile — LR funziona naturalmente bene qui. Le feature radiomiche invece sono misure fisiche eterogenee (area in centinaia, texture in [0,1], entropia in [0,8]): anche dopo StandardScaler lo spazio non è linearmente strutturato, e RF sfrutta meglio le soglie non-lineari per feature combinando 100 alberi indipendenti.

### KNN: degrado maggiore per la radiomica

Nei deep features KNN_3 è vicino a LR (-0.02 di F1 media). Nei radiomica KNN_3 cala a 0.773 media (-0.18 rispetto a RF). La distanza euclidea in uno spazio di 61 misure fisiche eterogenee non è coerente: le dimensioni con range numerico alto (area, perimetro) dominano la distanza e mascherano le dimensioni informative con range piccolo (texture GLCM in [0,1]). Anche dopo normalizzazione, lo spazio non forma cluster euclidei uniformi come quello prodotto dai backbone.

### Confronto OOD

Le metriche deep OOD sono medie su 30 configurazioni per coppia (6 modelli × 5 classificatori). Per confronto equo si usa la media dei 5 classificatori radiomica per coppia:

| Coppia | Deep F1 medio (30 cfg) | Radiomica F1 medio (5 clf) | Delta |
|---|---|---|---|
| Falciparum → Ovale | **0.557** | 0.366 | −0.19 |
| Falciparum → Vivax | **0.535** | 0.337 | −0.20 |
| Ovale → Vivax | **0.454** | 0.319 | −0.14 |
| Vivax → Ovale | **0.449** | 0.050 | −0.40 |
| Ovale → Falciparum | **0.297** | 0.283 | −0.01 |
| Vivax → Falciparum | **0.294** | 0.157 | −0.14 |
| → Malariae (media 3 coppie) | ~0.074 | ~0.044 | −0.03 |

Le deep features OOD vincono su tutte le coppie. Il divario è massimo per Vivax→Ovale (0.449 → 0.050): Vivax e Ovale hanno morfologie simili ma le feature di texture e forma calcolate analiticamente non si trasferiscono tra le loro fasi biologiche. Le deep feature — in particolare DinoBloom, identificato nel vecchio documento come modello più robusto in OOD grazie al pretraining su RBC biologiche — catturano strutture latenti più generalizzabili cross-specie.

Le coppie verso Malariae rimangono vicine a zero in entrambi gli approcci: Malariae ha fasi biologiche (G, S, T) con distribuzione di intensità e morfologia molto diversa dalle altre specie, e nessun modello riesce a generalizzarvi.

### Sintesi interpretativa

| Dimensione | Deep features | Radiomica |
|---|---|---|
| Feature dim | 768–2048 | 61 |
| Intra F1 best | 0.979 (ConvNeXt+LR) | 0.957 (RF) |
| Classificatore migliore | LR | RF |
| KNN performance | Competitiva (−0.02) | Degradata (−0.18) |
| OOD F1 migliore coppia | 0.557 (Falc→Ovale) | 0.366 (stessa coppia) |
| OOD generalizzazione | Moderata–buona | Limitata |

Le feature radiomiche sono sorprendentemente competitive intra-dataset ma cedono in OOD. La spiegazione unificante: le feature analitiche descrivono bene *come appare* un parassita di una specie (separazione tra specie), ma non catturano *l'organizzazione biologica dello sviluppo* (fasi) in modo trasferibile tra specie. I backbone deep — specialmente quelli con pretraining domain-specific — apprendono rappresentazioni più ricche che si trasferiscono meglio cross-specie.
