# Pacman - Capture the Flag: _despookjes_

Deze repo dient als inzending voor het toernooi "Pacman - Capture the Flag".
In `my_team.py` werd(en) de klasse(n) geïmplementeerd waarmee een team van twee agents geïnstantieerd kan worden in het `capture-the-flag` framework.

## 📥 Inzending

### Studiedeel
- **Opleiding:** [Bachelor in de artificiële intelligentie](https://caliweb.vub.be/?page=program&id=00721)
- **Studiedeel:** [AI Programmeerproject](https://caliweb.vub.be/?page=course-offers&id=011970)
- **Academiejaar:** _2025-2026_
- **Onderwijsteam:** Lynn Houthuys, Arno Temmerman

### Groepsleden
- Lisa Wei - lisa.wei@vub.be - vsi7-m
- Ralph Renoirte - ralph.renoirte@vub.be - ralphrenoirte


## 📚 Documentatie
Om dit team van agents uit te proberen in een spel "Capture the Flag" moet je in eerste instantie het `capture-the-flag` framework gedownload hebben van Canvas of van volgende [GitHub repo]().
Vervolgens raden we aan om de huidige repo (degene waar je nu de README van leest) te clonen/downloaden en als "team-map" te plaatsen in de `agents/` directory van het framwork.

```
capture-the-flag/
├─ agents/
│  └─ despookjes/
│     ├─ my_team.py
│     └─ README.md
⋮
├─ capture.py
⋮
└─ VERSION
```

Vervolgens kan je vanuit de `capture-the-flag` directory jouw agents (bijvoorbeeld als het rode team) laten spelen:
```bash
python capture.py -r agents/despookjes/my_team.py
```