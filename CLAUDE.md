# CLAUDE.md — Progetto tirocinio: classificazione malaria (Fase 6)

## Ambiente

- OS: Windows 10, Python 3.12.10 (non aggiornare a 3.14: PyTorch non è compatibile)
- GPU: NVIDIA GTX 1060 6 GB — VRAM è il vincolo principale
- CUDA: 12.6 | torch 2.6.0+cu124 già installato e funzionante
- Virtualenv: `.venv\Scripts\Activate.ps1` (non usare pip install senza attivarlo)
- Tutte le librerie necessarie sono già installate (timm, transformers, peft, albumentations, ecc.)

## Struttura del progetto

- Root: `d:\DESKTOP\TIROCINIO\malaria-detection-pipeline\`
- Script Fase 6: `scripts\phase 6\` (c'è uno spazio nel nome)
- CSV degli split: `csvs\kfold_heldout\` (fold train/val) e `csvs\splits_heldout\` (test held-out)
- Crop delle immagini: `Falciparum\`, `Malariae\`, `Ovale\`, `Vivax\` (ognuna con sottocartella `crops\`)

## Vincoli fissi — non modificare senza ok esplicito

- **Non rigenerare i CSV degli split**: gli split sono patient-aware e confrontabili con i risultati di Fase 5. Toccarli invalida il confronto.
- **Solo fold 1 e 2 sono informativi**: Ovale e Vivax hanno solo 2 group_id ciascuno; i fold 3-5 hanno validation set incompleto (mancano classi).
- **Mappatura classi fissa** in tutto il progetto:
  `SPECIES_TO_ID = {"Falciparum": 0, "Malariae": 1, "Ovale": 2, "Vivax": 3}`

## Decisioni già prese e implementate

- Path assoluti nei CSV risolti a runtime in `MalariaDataset.__init__` (vedi WORKLOG_FASE6.md)
- Il backbone va messo in `.eval()` durante il training in modalità `head_only`/`lora` (fix BatchNorm già presente in `trainer.py`)

## Diario di lavoro

Vedi `WORKLOG_FASE6.md` per: modifiche al codice con ragionamento, risultati dei run, problemi incontrati, decisioni metodologiche.
