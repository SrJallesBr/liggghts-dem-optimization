# Correção do painel e auditoria

O CSV `status_execucao.csv` depende do laço do controlador. Em `_run_doe_async`, a distribuição de um novo ensaio termina em `continue`, antes da atualização do painel. Portanto esse CSV não é um observador independente; não deve ser usado para concluir que a campanha pausou. A sequência exata ocorrida no PC empresarial não foi reproduzida localmente.

A correção é `runtime/monitor_campaign.py`: painel independente, somente leitura, sem NumPy. Lê `status.csv`, os diários DOE/refinamento, `results.csv` e as caudas dos logs. Tolera a última linha incompleta e ignora o CSV antigo. Mostra concluídos, rejeições/falhas, validade sinalizada por ensaio/etapa e aceites registrados, sem confundir validade com alvo atingido. Pendência registrada não confirma processo vivo; passo de log não é tempo físico.

## Aplicar à campanha ativa

Extrair APENAS o ZIP de correção na raiz da campanha. Ele contém o novo monitor e esta instrução; não sobrescreve controlador, configuração, contratos, ensaios nem resultados. Em outra aba WSL:

```bash
cd /mnt/c/DEM/DEM_empresa20_fresh_FINITE
python3 runtime/monitor_campaign.py
```

Para ler uma única vez, acrescentar `--once`. Ctrl+C fecha apenas o painel. Não executar `--start` novamente. Não substituir a campanha ativa pelo ZIP completo.

## Auditoria e Git

Antes desta correção, os 74 arquivos do diretório de desenvolvimento eram idênticos, byte a byte, ao ZIP completo entregue. Esse estado foi registrado no commit-base `eca067e`. A correção acrescenta monitor, regressões, documentação e exclusões Git; não altera arquivos do solver, física, configuração, autorização nem implementação do otimizador. Os 20 workers permanecem.

Também foi inspecionado o diff contra `d00dbf1` do repositório de continuação: configuração fresh, refinamento limitado, autorização isolada, admissão e recibos científicos. Esse repositório tem alterações de outro trabalho e não foi limpo nem sobrescrito. A versão fresh agora possui Git próprio, local, sem remoto configurado.

Não foi identificado um desvio comprovadamente indevido a restaurar nesta revisão. Não foi feito rollback indiscriminado: reverter a uma campanha anterior mudaria o escopo fresh autorizado. Isso não é certificação física nem auditoria completa de todo o histórico anterior. A instalação e o funcionamento no PC empresarial ainda precisam de confirmação.
