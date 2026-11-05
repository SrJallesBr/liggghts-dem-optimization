# Versão oficial — paralelismo contínuo

Esta é a versão oficial do projeto `SrJallesBr/liggghts-dem-optimization`,
na branch principal `main`. Substitui o código antigo da branch principal.

## Escopo

- Base funcional: commit `709dce8`, com monitor separado e paralelismo contínuo.
- 20 workers, DOE com alvo de 84 válidos e teto de 420 tentativas;
  até 748 tentativas de refinamento, respeitando 1000 tentativas globais.
- DOE em voo é preservado na transição; refinamentos ocupam as vagas livres.
- Entrada oficial: `runtime/run_fresh_finite.py`; instruções no `README.md`.
- Esta promoção ajusta documentação, manifesto de integridade e sua âncora.
  Não altera física, configuração, critérios de admissão ou lógica do solver.

## Limites mantidos

Oficial designa a versão de software escolhida pelo responsável pelo projeto.
Não significa calibração física validada, aprovação de fornecedor, nem
certificação do ambiente da empresa. `validation_trials=0`: não há ensaios
independentes adicionais de confirmação configurados.

As menções a dados sintéticos descrevem exclusivamente fixtures dos testes.
As menções a candidato descrevem o resultado da busca de parâmetros, não o
status da versão. Nomes internos e arquivos de proveniência são preservados
para não quebrar referências nem contornar guardas.

## Atualização e recuperação

Não substitua arquivos em campanhas ativas. Para nova campanha, use uma pasta
nova. Para retomar uma campanha existente, preserve a pasta inteira, aguarde
uma janela segura, reconcilie os casos e execute o check antes de iniciar.
Não importe resultados de outra campanha nem remova locks/guardas para forçar
uma retomada.

O histórico anterior da `main` foi preservado na branch
`backup/main-antes-oficial` e no histórico Git da promoção. A publicação no
GitHub não aplica esta versão a nenhuma campanha em execução.
