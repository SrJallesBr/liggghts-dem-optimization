# Verificação — paralelismo contínuo

Suíte completa no WSL: 32 testes passaram em 330,136 s, exit code 0.

Comando: `OPENBLAS_NUM_THREADS=1 python3 -m unittest discover -s tests -p 'test*.py' -v`.

Inclui sete novas regressões com eventos de sincronização: 83 válidos e 20 workers ocupados; reposição de rejeição; refinamento antes do término dos DOE lentos; aproveitamento dos DOE tardios nas novas propostas sem linhas duplicadas; teto DOE; capacidade global incluindo reservas em voo; drenagem por erro técnico e por candidato aceito. Os cenários de drenagem compartilham testes, portanto essa lista não é uma contagem de métodos.

As regressões existentes foram ajustadas para permitir DOE extras em voo, mantendo retomada sem duplicatas. Também passaram admissão científica positiva/negativa, recusa de artefatos adulterados, recusa de respostas sem recibo, proposta do surrogate real sobre dados sintéticos, monitor somente leitura e subprocesso Python de teste através das guardas reais. Nenhum solver DEM de produção foi iniciado.

A primeira execução das três novas regressões falhou na versão antiga. A primeira suíte ampliada detectou uma asserção histórica que exigia exatamente 94 tentativas após dez rejeições; ela foi substituída por verificação de contagem dentro do limite de concorrência, validade dos resultados e ausência de duplicatas. A suíte completa posterior passou.

Os testes não validam física, calibração, desempenho, solver ou hardware do computador empresarial. O preflight de 20 CPUs é uma fixture explícita; não comprova capacidade de outra máquina. A campanha ativa não foi alterada nem reiniciada.

Os registros anteriores em VERIFICACAO.md e VERIFICACAO_MONITOR.md são históricos. Esta revisão é descrita em PARALELISMO_CONTINUO.md.
