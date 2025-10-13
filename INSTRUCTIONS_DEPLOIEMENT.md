# Instructions pour déployer l'application sur Render

Bonjour ! Voici le guide étape par étape pour mettre en ligne votre application d'analyse de football.

**Prérequis :**
- Un compte GitHub (gratuit)
- Un compte Render (le plan gratuit est suffisant)

---

### Étape 1 : Mettre le code sur votre GitHub

Render fonctionne en se connectant à un dépôt de code GitHub. Vous devez donc d'abord y placer le projet.

1.  **Créez un nouveau dépôt sur GitHub :**
    *   Allez sur [github.com/new](https://github.com/new).
    *   Donnez-lui un nom (par exemple, `analyse-football-app`).
    *   Choisissez "Public" ou "Privé" selon votre préférence.
    *   Cliquez sur **"Create repository"**.

2.  **Téléversez tous les fichiers du projet :**
    *   Sur la page de votre nouveau dépôt, cliquez sur **"Add file"** > **"Upload files"**.
    *   Faites glisser **tous les fichiers et dossiers** du projet dans la zone de téléversement.
    *   Validez en cliquant sur **"Commit changes"**.

---

### Étape 2 : Créer le service web sur Render

1.  **Connectez-vous à votre compte Render.**
2.  **Allez sur votre "Dashboard"** (tableau de bord).
3.  **Cliquez sur "New +"** puis sur **"Web Service"**.
4.  **Connectez votre compte GitHub à Render** si ce n'est pas déjà fait.
5.  **Sélectionnez votre dépôt** (celui que vous venez de créer, ex: `analyse-football-app`).
6.  **Remplissez les informations du service :**
    *   **Name** : Donnez un nom à votre application (ex: `analyse-football`). L'URL de votre site sera `https-nom-choisi.onrender.com`.
    *   **Region** : Laissez la valeur par défaut (Frankfurt est un bon choix pour l'Europe).
    *   **Branch** : Laissez `main` (ou le nom de la branche principale de votre dépôt).
    *   **Root Directory** : Laissez vide.
    *   **Runtime** : Choisissez **`Python 3`**.
    *   **Build Command** : Copiez-collez exactement ceci : `pip install -r requirements.txt`
    *   **Start Command** : Copiez-collez exactement ceci : `./start.sh`

7.  **Choisissez le plan gratuit ("Free")** sous la section "Instance Type".

8.  **Développez la section "Advanced Settings" (Paramètres avancés) :**
    *   Cliquez sur **"Add Environment Variable"**.
    *   Dans le champ **`Key`**, mettez `PYTHON_VERSION`.
    *   Dans le champ **`Value`**, mettez `3.9.9` (ou une version récente de Python 3.9).

9.  **Cliquez sur "Create Web Service"**.

---

### Étape 3 : Attendre et lancer l'application

*   Render va maintenant commencer à construire et à déployer votre application. Cela peut prendre 5 à 10 minutes la première fois. Vous verrez les logs de construction s'afficher.
*   Une fois que vous voyez le message **"Your service is live"**, le déploiement est terminé.
*   Vous pouvez accéder à votre application en cliquant sur le lien en haut de la page (ex: `https://analyse-football.onrender.com`).

Et voilà ! Votre application sera en ligne et prête à être utilisée.

**Note importante sur le plan gratuit :**
*   Le service se met en veille après 15 minutes d'inactivité. Le premier chargement après une période de veille peut donc prendre 30 à 60 secondes. C'est normal.
*   L'analyse de vidéos longues peut être lente ou dépasser les limites de mémoire du plan gratuit. C'est idéal pour des clips plus courts.

Si vous avez la moindre question, n'hésitez pas.