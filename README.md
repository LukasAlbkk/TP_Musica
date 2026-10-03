# Riffs de Rock 80/90 com Algoritmos Genéticos

TP1 — Composição Musical Algorítmica (DCC831, UFMG, 2026/2).
Lucas Albuquerque Santos Costa - 2023028005

Um algoritmo genético (AG) evolui riffs de guitarra no estilo **rock das
décadas de 1980-90**. O AG evolui apenas a melodia/riff principal; o
acompanhamento (power chords, baixo e bateria) é gerado por regras de
teoria musical a partir de progressões de acordes típicas do gênero, o que
fixa a restrição estilística. Cada execução produz um arquivo **MIDI**
(saída simbólica, reproduzível) e um **WAV** sintetizado apenas para
audição, além de um gráfico da evolução do fitness ao longo das gerações.

Veja `paper/summary.tex` para a descrição técnica completa (formato ISMIR).

## Instalação

Requer Python 3.10+.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

pip install -r requirements.txt
```

Nenhum software externo é necessário — o áudio é sintetizado por um
softsynth próprio (numpy), sem precisar de FluidSynth nem de soundfonts.

## Uso

### Gerar uma música

```bash
python scripts/generate.py --name minha_musica --seed 7 \
    --key-root A --key-mode natural_minor --progression i-VI-III-VII \
    --tempo 132 --mutation-rate 0.06
```

Saída em `outputs/demo/minha_musica.{mid,wav,json}` e
`minha_musica_fitness.png`. Rode `python scripts/generate.py --help` para
ver todos os hiperparâmetros configuráveis (tom, modo, progressão,
andamento, resolução rítmica, tamanho de população, número de gerações,
taxas de cruzamento/mutação, tamanho do torneio, elitismo, paciência do
early stopping, semente).

### Gerar o conjunto de demonstração (6 músicas)

```bash
python scripts/make_demo_set.py
```

Gera 6 músicas em `outputs/demo/`: duas com a *mesma configuração* e
sementes diferentes (demonstrando que o sistema produz músicas distintas a
partir da mesma configuração), e quatro variando exatamente um
hiperparâmetro cada (taxa de mutação, tom/escala/progressão, resolução
rítmica, andamento+progressão). Também escreve
`outputs/demo/fitness_comparison.png` (comparação do fitness entre as seis
execuções) e `outputs/demo/summary.csv`.

### Testes

```bash
python tests/test_smoke.py
```

### Áudio com melhor qualidade (opcional)

O softsynth embutido já produz áudio ouvível sem instalar nada. Para um
som mais realista (guitarra/bateria via General MIDI), instale
[FluidSynth](https://www.fluidsynth.org/) e baixe um soundfont livre (ex.:
`FluidR3_GM.sf2`), depois passe `--soundfont caminho/para/arquivo.sf2` para
`scripts/generate.py`.

## Estrutura do código

```
src/ga_music/
  theory.py       escalas, progressões de acordes de rock, instrumentos GM
  config.py       hiperparâmetros de uma execução (GAConfig)
  genome.py       representação do genótipo (grade REST/HOLD/nota) e decodificação
  fitness.py      função de fitness heurística (6 termos de teoria musical)
  ga.py           laço do algoritmo genético (seleção, cruzamento, mutação, elitismo)
  arrangement.py  monta a música completa (riff evoluído + acompanhamento fixo) em MIDI
  synth.py        sintetizador próprio (MIDI -> WAV), sem dependências externas
  report.py       gráficos de evolução do fitness
scripts/
  generate.py        gera uma música a partir de hiperparâmetros na linha de comando
  make_demo_set.py   gera o conjunto de 6 músicas de demonstração
paper/
  summary.tex        resumo no formato ISMIR (template oficial 2026)
outputs/demo/         MIDI, WAV, gráficos e logs das músicas de demonstração (versionados)
```
