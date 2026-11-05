# Paralelismo contínuo: DOE e refinamento

## Comportamento

- Mantém os 20 workers configurados ocupados com novas tentativas DOE enquanto a meta de 84 pontos DOE válidos não for atingida. O número de casos em andamento não é limitado ao número de válidos que falta.
- Uma rejeição libera a vaga para outra tentativa, mesmo quando falta apenas um válido.
- Ao atingir a meta e satisfazer a guarda de prontidão científica existente, não inicia mais DOE. Preserva todos os DOE em execução e usa as vagas livres para refinamentos, no MESMO executor de 20 workers.
- Cada DOE tardio concluído é registrado e seus dados válidos passam a integrar as próximas propostas do modelo. Não altera parâmetros de refinamentos já iniciados.
- Os pontos DOE extras são diferentes entre si. Seus IDs e parâmetros ficam reservados antes de iniciar o refinamento; não há colisão de IDs nem repetição de pontos em voo.
- A soma DOE + refinamentos em execução não ultrapassa 20. Isso é um limite de concorrência, não promessa de 100% de uso constante da CPU: proposta do modelo, preparação e I/O também usam tempo.

## Limites e segurança preservados

- Máximo de 420 tentativas DOE, até 748 tentativas de refinamento e 1000 tentativas globais. O orçamento disponível de refinamento desconta TODAS as reservas DOE, inclusive as ainda em execução.
- Rejeição científica não é erro técnico: rejeições recebem substitutos; erro técnico interrompe novos lançamentos e aguarda os casos em voo das duas fases terminarem.
- Ao encontrar candidato aceito ou esgotar o orçamento, aguarda os DOE/refinamentos já em execução; não os cancela.
- Critérios físicos, decks, fatores, discretização, geometria, carregamento, ângulos, guardas de proveniência, autorizações e configuração de 20 workers não foram alterados.
- O painel separado continua disponível com `python3 runtime/monitor_campaign.py`. O agendador misto também inclui DOE e refinamentos ativos no CSV de execução.

## Aplicação

Esta mudança NÃO altera o processo Python que já está em execução. NÃO extraia a atualização nem substitua módulos enquanto a campanha estiver rodando: há imports tardios, e misturar versões seria inseguro. Nenhuma campanha ativa foi parada ou reiniciada para produzir esta entrega.

A branch `main` contém a versão oficial completa, sem `run/`, estado ou resultados de campanhas. O ZIP parcial distribuído anteriormente é um artefato histórico de atualização, destinado exclusivamente à cópia `DEM_empresa20_fresh_FINITE` daquela entrega; não substitui a distribuição oficial completa. Não aplique esse ZIP a outras campanhas.

Quando houver uma janela segura, com controlador E solvers encerrados de forma controlada e os casos em voo reconciliados, faça backup da cópia completa antes de aplicar a atualização. Preserve `run/`, configuração, manifestos DOE e todos os resultados. Não remova reservas ou locks para forçar retomada.

Na MESMA pasta, com o venv WSL já utilizado pela campanha ativado:

```bash
python3 runtime/run_fresh_finite.py --check
```

O check não inicia simulações. Se acusar incompatibilidade, estado incerto ou processo ainda ativo, pare e investigue; não contorne a guarda.

Apenas após check aprovado e autorização para iniciar/retomar:

```bash
python3 runtime/run_fresh_finite.py --start
```

Em outra aba, se necessário:

```bash
python3 runtime/monitor_campaign.py
```

## Verificação de software

As regressões usam respostas sintéticas explicitamente marcadas e eventos de sincronização; não executam LIGGGHTS. Cobrem falta de um válido com 20 workers, reposição após rejeição, sobreposição DOE/refinamento, inclusão dos DOE tardios, ausência de duplicação dos dados do modelo, teto conjunto, limites de tentativas/orçamento, drenagem após candidato e erro técnico tardio. Isso não valida a física nem o desempenho do computador empresarial.
