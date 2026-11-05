# Verificação da correção de monitoramento

- Suíte completa no WSL: 25 testes aprovados em 315,074 s.
- Cinco regressões do monitor também aprovadas no Python do Windows.
- CLI executada com fixture temporária sintética: exibiu trial, etapa e passo do log.
- Hashes dos arquivos da fixture antes/depois iguais: leitura sem alterar entradas.
- Os 74 arquivos anteriores à correção foram comparados byte a byte com o ZIP entregue e registrados no commit-base eca067e.
- Nenhum solver de produção foi iniciado por esta revisão. Os testes de execução usam subprocessos/solvers de teste.
- Não houve acesso ao PC empresarial nem alteração/parada da campanha ali executada.
- Não houve validação física. Um ponto sinalizado válido pelo software continua distinto de alvo atingido e validação independente.

Reproduzir:

```bash
python3 -m unittest discover -s tests -p 'test*.py' -v
python3 runtime/monitor_campaign.py --run-root /caminho/da/campanha/run --once
```
