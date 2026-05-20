  /* ── Helpers ── */
  function getApiKey() {
    return document.getElementById('api-key').value;
  }
 
  function showResult(boxId, iconId, probaId, labelId, data) {
    const probaStr = data.probabilite_de_depart;
    const probaNum = parseInt(probaStr);
    const depart = probaNum > 30 ? 1 : 0;
 
    const box = document.getElementById(boxId);
    box.className = `result-box col-3 ${depart ? 'depart' : 'reste'}`;
    document.getElementById(iconId).textContent = depart ? '⚠️' : '✅';
    document.getElementById(probaId).textContent = probaStr;
    document.getElementById(labelId).textContent = data.resultat || ' de chances de quitter l\'entreprise';
    box.style.display = 'block';
  }
 
  /* ── Prédiction par ID ── */
  async function predictById() {
    const id       = document.getElementById('employee-id').value;
    const apiKey   = getApiKey();
    const resultBox = document.getElementById('result-id');
    const errorBox  = document.getElementById('error-id');
    const spinner   = document.getElementById('spinner-id');
 
    errorBox.style.display = 'none';
    resultBox.style.display = 'none';
 
    if (!id)     { errorBox.textContent = "Veuillez saisir un ID employé."; errorBox.style.display = 'block'; return; }
    if (!apiKey) { errorBox.textContent = "Veuillez saisir votre clé API."; errorBox.style.display = 'block'; return; }
 
    spinner.style.display = 'block';
    try {
      const response = await fetch(`/predict/${id}`, {
        method: 'POST',
        headers: { 'X-API-Key': apiKey }
      });
      if (!response.ok) { const e = await response.json(); throw new Error(e.detail || `Erreur ${response.status}`); }
      const data = await response.json();
      if (data.Erreur) { errorBox.textContent = data.Erreur; errorBox.style.display = 'block'; return; }
      showResult('result-id', 'icon-id', 'proba-id', 'label-id', data);
    } catch (err) {
      errorBox.textContent = err.message || "Erreur inattendue.";
      errorBox.style.display = 'block';
    } finally {
      spinner.style.display = 'none';
    }
  }
 
  /* ── Prédiction par formulaire ── */
  async function predictByForm() {
    const apiKey   = getApiKey();
    const resultBox = document.getElementById('result-form');
    const errorBox  = document.getElementById('error-form');
    const spinner   = document.getElementById('spinner-form');
 
    errorBox.style.display = 'none';
    resultBox.style.display = 'none';
 
    if (!apiKey) { errorBox.textContent = "Veuillez saisir votre clé API."; errorBox.style.display = 'block'; return; }
 
    /* Lecture des champs */
    const fields = [
      'heure_supplementaires', 'age', 'genre', 'revenu_mensuel', 'poste',
      'nombre_experiences_precedentes', 'annee_experience_totale',
      'annees_dans_l_entreprise', 'annees_dans_le_poste_actuel',
      'nombre_participation_pee', 'nb_formations_suivies',
      'distance_domicile_travail', 'niveau_education', 'frequence_deplacement',
      'annees_depuis_la_derniere_promotion', 'annes_sous_responsable_actuel',
      'departement', 'augmentation_salaire_precedente_pourcentage'
    ];
 
    const strFields = new Set(['poste', 'frequence_deplacement', 'departement']);
    const payload   = {};
    let missing     = false;
 
    for (const f of fields) {
      const el = document.getElementById(`f-${f}`);
      if (!el || el.value === '') {
        errorBox.textContent = `Le champ « ${f} » est requis.`;
        errorBox.style.display = 'block';
        missing = true;
        break;
      }
      payload[f] = strFields.has(f) ? el.value : parseInt(el.value);
    }
    if (missing) return;
 
    spinner.style.display = 'block';
    try {
      const response = await fetch('/predict_nouveau', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-API-Key': apiKey
        },
        body: JSON.stringify(payload)
      });
      if (!response.ok) { const e = await response.json(); throw new Error(e.detail || `Erreur ${response.status}`); }
      const data = await response.json();
      if (data.Erreur) { errorBox.textContent = data.Erreur; errorBox.style.display = 'block'; return; }
      showResult('result-form', 'icon-form', 'proba-form', 'label-form', data);
    } catch (err) {
      errorBox.textContent = err.message || "Erreur inattendue.";
      errorBox.style.display = 'block';
    } finally {
      spinner.style.display = 'none';
    }
  }

    /* ── Prédiction par formulaire ── */
  async function predictByPoste() {
    const apiKey    = getApiKey();
    const resultBox = document.getElementById('result-poste');
    const errorBox  = document.getElementById('error-poste');
    const spinner   = document.getElementById('spinner-poste');
    const payload   = {};

    errorBox.style.display = 'none';
    resultBox.style.display = 'none';

    if (!apiKey) { errorBox.textContent = "Veuillez saisir votre clé API."; errorBox.style.display = 'block'; return; }

    payload["poste"] = document.getElementById('p-poste').value;

    spinner.style.display = 'block';
    try {
      const response = await fetch('/predict_poste', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-API-Key': apiKey },
        body: JSON.stringify(payload)
      });
      if (!response.ok) { const e = await response.json(); throw new Error(e.detail || `Erreur ${response.status}`); }
      const data = await response.json();
      if (data.Erreur) { errorBox.textContent = data.Erreur; errorBox.style.display = 'block'; return; }

      // Remplir les stats
      document.getElementById('stat-total').textContent   = data.nombre_employes;
      document.getElementById('stat-departs').textContent = data.nombre_departs_prevus;
      document.getElementById('stat-taux').textContent    = data.taux_de_depart_prevu;
      document.getElementById('stat-risque').textContent  = data.taux_de_risque_moyen;

      // Construire le top 5 dynamiquement
      const container = document.getElementById('top5-container');
      container.innerHTML = '';
      data.top5_employes_a_risque.forEach((e, i) => {
        const proba = parseInt(e.probabilite_de_depart);
        const color = proba >= 80 ? '#c0392b' : proba >= 30 ? '#e67e22' : '#1a6fb5';
        container.innerHTML += `
          <div class="col-md-12">
            <div style="display:flex; align-items:center; justify-content:space-between;
                        background: ${color}; border: 1px solid #c5dff5;
                        border-radius: 10px; padding: 0.8rem 1.2rem;">
              <div style="display:flex; align-items:center; gap: 12px;">
                <span style="background:black; color:#fff; border-radius:50%;
                            width:28px; height:28px; display:flex; align-items:center;
                            justify-content:center; font-size:0.8rem; font-weight:600;">${i + 1}</span>
                <span style="font-weight: 600; color: black;">Employé #${e.id_employee}</span>
              </div>
              <span style="font-size: 1.1rem; font-weight: 700; color: black;">
                ${e.probabilite_de_depart}
              </span>
            </div>
          </div>`;
      });

      resultBox.style.display = 'block';
    } catch (err) {
      errorBox.textContent = err.message || "Erreur inattendue.";
      errorBox.style.display = 'block';
    } finally {
      spinner.style.display = 'none';
    }
  }
 
  /* ── Entrée clavier ── */
  document.getElementById('employee-id').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') predictById();
  });