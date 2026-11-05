"""Trial-local material provenance and compact image labels (no solver calls)."""
from pathlib import Path
import hashlib


def properties_from_deck(snapshot, kind):
    from optimizer.rocky_io import _variables, evaluate
    if kind == 'rolling':
        deck = Path(snapshot).parent / 'in.rolamento'
    else:
        deck=Path(snapshot).parents[1]/('in.repouso' if kind=='repose' else 'in.deslizamento')
    payload=deck.read_bytes();variables=_variables(payload.decode('utf-8'))
    keys=['sf11','rf11','CED11','density','youngmodulus1','poisson1','CoR11',
          'sf12','sf13','rf12','rf13','CED12','CED13','nbrratio']
    result={key:evaluate('v_'+key,variables) for key in keys if key in variables}
    result.update(source_deck=str(deck),source_deck_sha256=hashlib.sha256(payload).hexdigest(),
                  unavailable_keys=[key for key in keys if key not in variables])
    return result


def image_label(result):
    p=result.get('material_properties',{})
    def value(key):
        return f'{p[key]:.6g}' if key in p else 'indisponível'
    material=(f"sf11={value('sf11')} | rf11={value('rf11')} | CED11={value('CED11')} J/m³\n"
              f"ρ={value('density')} kg/m³ | E={value('youngmodulus1')} Pa | ν={value('poisson1')} | CoR11={value('CoR11')}\n"
              f"sf12/13={value('sf12')}/{value('sf13')} | rf12/13={value('rf12')}/{value('rf13')} | CED12/13={value('CED12')}/{value('CED13')} J/m³")
    diagnosis=result.get('geometry_diagnosis') or {}
    policy_enabled=result.get('config_used',{}).get('response_policy')=='user-unmeasurable-as-90-v1'
    if result.get('status')!='measured' and not policy_enabled:
        state=('COLUNA RETIDA | ' if diagnosis.get('detected') else '')+'Sem resposta adotada; política de 90° desabilitada'
    elif diagnosis.get('detected'):
        state='COLUNA RETIDA: 90° adotados; não é ângulo medido nem causa de coesão comprovada'
    elif result.get('status')!='measured':
        state='SEM ÂNGULO ACEITO: 90° somente pela convenção habilitada; não medidos'
    else:
        state='Ajuste aceito pelo critério local; não é validação física'
    criterion=result.get('minimum_fit_r_squared')
    quality='' if criterion is None else f' | R² mínimo={criterion:.2f}'
    source=Path(result['source_path']) if result.get('source_path') else None
    identity='' if source is None else source.parents[3].name+' / '+source.parents[2].name+'\n'
    return identity+material+'\n'+state+quality
