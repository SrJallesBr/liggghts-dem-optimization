# Otimizador DEM - DOE, superficie de resposta e refinamento

Esta versao preserva o ensaio LIGGGHTS. Ela roda 30 pontos iniciais em Latin
Hypercube, ajusta uma superficie quadratica de 15 termos e seleciona os testes
seguintes pelo melhor resultado previsto. O arquivo ANALISE_DOE.json mostra
erro de validacao, melhor ponto observado e sensibilidades locais.

Pacote portatil para WSL/Linux. Nao requer NumPy nem instalacao Python adicional.

## O que ele otimiza

- `sf11`: atrito deslizante particula-particula.
- `rf11`: atrito de rolamento particula-particula.
- `rf12`: atrito de rolamento particula-superficie tipo 2 (as bases).
- `CED11`: energia de coesao particula-particula.

O modelo usa os snapshots `post/angulo_repouso.vtk` e `post/drawdown_final.vtk` para medir automaticamente os dois angulos. A busca alterna grupos: `rf11`/`rf12` para priorizar o repouso e `sf11`/`CED11` para priorizar o drawdown. A aprovacao final continua exigindo que os dois alvos estejam dentro da tolerancia.

## Antes de rodar

1. Copie esta pasta para o WSL, por exemplo `~/dem_optimization`.
2. Confirme que `liggghts` e `mpirun` existem no WSL:

```bash
liggghts -h
mpirun --version
```

3. Edite `configuracao_otimizacao.txt` e defina:
   - `target_repose_deg`
   - `target_drawdown_deg`
   - `tolerance_deg`
   - os limites de `sf11`, `rf11` e `CED11`
   - `parallel_trials` conforme a RAM disponivel.

## Rodar

No Windows, use `Abrir-Otimizacao-DEM.cmd`. Ele abre uma aba visivel do
Windows Terminal com a saida do controlador e a mantem aberta quando o
processo termina ou falha.

Para uma execucao manual dentro do WSL:

```bash
cd ~/dem_optimization
chmod +x rodar_otimizacao_wsl.sh
bash rodar_otimizacao_wsl.sh
```

## Acompanhar

```bash
tail -f controlador.log
cat resultados.csv
free -h
```

O resultado final fica em `MELHOR_RESULTADO.json`. A busca so e considerada convergida quando os dois angulos estiverem dentro da tolerancia configurada.

## Seguranca do caso

O arquivo original nao e alterado. Cada tentativa e criada em `trials/trial_###`, com seus VTKs e log proprios.

