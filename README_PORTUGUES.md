# DEM empresa — campanha nova, finita e independente

Este pacote substitui a preparação anterior. O ponto de entrada é somente
`runtime/run_fresh_finite.py`. Extraia em uma pasta NOVA, fora da campanha
antiga; não sobreponha arquivos. Nenhum resultado, journal, autorização ou
estado de execução da campanha antiga é incluído/importado. Os modelos,
malhas e o empacotamento inicial estático são os insumos, não resultados DOE.

## Executar no ambiente Linux/WSL que já funciona na empresa

Abra o terminal do ambiente existente e entre na pasta extraída. Use o Python
desse ambiente (nos exemplos, `python3`). Não é preciso reinstalar o ambiente.

Verificar sem lançar qualquer solver:

```sh
python3 runtime/run_fresh_finite.py --check
```

Iniciar a campanha completa, ou retomar a MESMA pasta depois de um término
controlado:

```sh
python3 runtime/run_fresh_finite.py --start
```

O padrão é `/usr/local/bin/liggghts`. Se o binário funcional estiver em outro
lugar, passe seu caminho verdadeiro, sem editar configuração nem código:

```sh
python3 runtime/run_fresh_finite.py --check --solver /caminho/existente/liggghts
python3 runtime/run_fresh_finite.py --start --solver /caminho/existente/liggghts
```

Os exemplos são comandos LINUX/WSL, não Python nativo do Windows. O pacote
exige que o WSL/Python enxergue pelo menos 20 CPUs. Não altera configurações do
Windows, WSL ou da empresa para obter esses recursos. O `--check` verifica
arquivos, configuração, CPUs e presença do executável; não executa uma análise
DEM e não certifica compatibilidade física do solver.

## Política implementada

- 20 ensaios simultâneos, um rank por ensaio; cada ensaio executa seus testes
  sequencialmente. Threads auxiliares numéricas limitadas a uma.
- IDs novos a partir de 1. DOE inicial de 84 pontos distintos, alvo de 84 pontos
  completos e admitidos pelos medidores. Rejeições recebem backfill de novos
  pontos, com teto de 420 tentativas DOE; não há repetição ilimitada.
- O refinamento começa somente depois de satisfeito o DOE e sua verificação
  de prontidão. Reutiliza o refinador/modelo substituto finito do projeto.
  Enquanto faltar DOE válido, mantém a ocupação dos 20 workers; ao atingir o
  requisito, preserva DOE em voo e preenche as vagas livres com refinamentos.
  Veja `PARALELISMO_CONTINUO.md` para aplicação segura e limites da atualização.
- Até 748 tentativas de refinamento, reduzidas se necessário para não superar
  1000 tentativas totais DOE + refinamento. Ex.: com 420 tentativas DOE restam
  no máximo 580 de refinamento. Os tetos não são metas obrigatórias.
- O fluxo pode terminar antes se encontrar candidato dentro das tolerâncias.
  Candidato encontrado NÃO é uma confirmação independente da calibração.
  Não há ensaios independentes adicionais configurados nesta versão
  (`validation_trials=0`).
- Erro técnico impede novas admissões; trabalhos já iniciados são drenados.
  Se o DOE não atingir o requisito, encerra como `doe_insufficient`, sem
  refinar nem fabricar ângulos para preencher o alvo.
- Journals persistentes impedem relançamento de tentativas concluídas. Depois
  de interrupção abrupta, arquivos incompletos/processos órfãos podem exigir
  reconciliação; o programa bloqueia em vez de assumir sucesso ou duplicar.

## Isolamento científico do repouso

O protocolo de repouso atual permanece separado dos ângulos históricos. Há
pins de integridade do pacote, das entradas e da implementação, configuração
fixada e checagem de propriedade dos IDs/workspaces. A admissão de medição
continua exigindo o recibo de proveniência do controlador e correspondência
com o ensaio local; trocar rótulo ou número de ensaio não valida dados antigos.
A liberação é apenas para este pacote independente e seu lançamento explícito.
A campanha original e sua guarda não são alteradas. O metadado histórico do
overlay permanece como proveniência dos modelos, não como estado desta execução.

Não copie resultados antigos para `run/`, não substitua os modelos pelo meio
da campanha e não edite os arquivos pinados. Essas ações invalidam a admissão.
Um novo experimento físico requer novo pacote revisado, não remover a guarda.

## Saídas e testes

A pasta `run/` nasce na execução. Guarda resultados CSV, journals DOE/adaptativo,
subpastas `trial_XXXX`, relatórios e `finite_summary.json`. Guarde a pasta inteira
para retomar. O ZIP entregue não contém essa pasta nem resultados sintéticos.

Testes offline reproduzíveis, sem lançar o solver de produção:

```sh
python3 -m unittest discover -s tests -p 'test_fresh*.py' -v
```

Os testes usam respostas sintéticas e um subprocesso-fixture explicitamente
identificado, além de exercitar o modelo substituto real. Não comprovam
ângulos, estabilidade do material ou funcionamento no PC da empresa. Consulte
`VERIFICACAO.md` para o resultado da execução dos testes e seus limites.

Os modelos físicos existentes foram preservados. Não foi executada campanha
DEM de produção nem preparada/validada a instalação da empresa.
