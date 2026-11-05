# Verificação de software

## Resultado real

Executado no WSL Ubuntu-24.04, Python 3.12.3:

```sh
python3 -m unittest discover -s tests -p 'test_fresh*.py' -v
```

Resultado: **20 testes passaram**, `Ran 20 tests in 288.350s`, `OK`,
código de saída 0.

Cobertura exercitada:
- 84 pontos iniciais distintos e orçamento configurado;
- backfill: 10 rejeições sintéticas + 84 respostas admitidas em 94 tentativas;
- teto de 420 rejeições, sem iniciar refinamento com DOE insuficiente;
- redução do orçamento de refinamento para 580 depois de 420 tentativas DOE;
- 748 tentativas sintéticas de refinamento, 20 workers e retomada sem duplicatas;
- transição integrada DOE -> refinamento;
- proposta do modelo substituto real sobre DOE com respostas sintéticas;
- erro técnico impede admissão de novas tentativas;
- preparo real do ensaio ID 1 usando os modelos atuais, sem lançar solver;
- CLI --check sem lançamento de solver (capacidade de 20 CPUs simulada SOMENTE
  dentro desse teste);
- admissão científica positiva da cópia independente e recusa de adulteração;
- configuração em memória diferente da pinada é recusada;
- ausência de recibo de medição não é evidência para ajuste;
- histórico/workspace não pertencente à campanha é recusado;
- subprocesso Python-fixture através das guardas reais de lançamento. Esse
  subprocesso apenas imprime `SYNTHETIC_SOLVER_FIXTURE`; não é o solver DEM.

## Limites explícitos

Os cenários extensos de agendamento substituem o executor de ensaios por
respostas sintéticas. Nesses testes, somente a admissão dessas respostas é
substituída; a admissão real é exercitada separadamente pelos testes das
barreiras. Não apresentar esses resultados como ângulos DEM ou validação física.

O comando --check real, sem simulação da contagem de CPUs, foi executado:

```
preflight failed: parallel_trials=20 exceeds CPUs visible to Python/WSL (12)
```

Essa recusa é esperada neste WSL local e confirma que a guarda não foi removida.
O teste com capacidade simulada NÃO valida hardware de 20 CPUs. O PC da empresa,
sua instalação e seu solver não foram inspecionados nem certificados.

Nenhuma campanha DEM de produção foi executada para preparar este pacote.
Nenhum dado/estado da campanha antiga foi importado. Os testes não comprovaram
estabilidade, concordância com ANSYS ou acerto dos alvos físicos. O encontro de
um candidato não equivale a ensaios independentes adicionais de confirmação;
`validation_trials=0` nesta configuração.

## Ambiente usado pelos testes

- Python 3.12.3
- NumPy 2.5.4; SciPy 1.18.1
- Matplotlib 3.11.2; pytest 9.1.1
- CairoSVG 2.9.1; Pillow 13.0.0

O arquivo de requisitos documenta o ambiente de referência; não é uma ordem
para reinstalar ou modificar o ambiente funcional da empresa.
