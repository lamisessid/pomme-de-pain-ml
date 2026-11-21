def test_health(client):
    r = client.get("/health").json()
    assert r["status"] == "ok"
    assert r["model_loaded"] is True
    assert r["products"] == 50


def test_dashboard(client):
    r = client.get("/api/v1/dashboard").json()
    assert r["kpis"]["sales"] > 0
    assert len(r["trend"]) == 30
    assert len(r["top_products"]) == 5
    assert r["tomorrow"]["predicted_sales"] > 0


def test_dashboard_out_of_range(client):
    assert client.get("/api/v1/dashboard", params={"day": "2030-01-01"}).status_code == 404


def test_forecasts(client):
    r = client.get("/api/v1/forecasts", params={"days": 7}).json()
    assert len(r["daily"]) == 7
    assert len(r["by_product"]) == 50
    for d in r["daily"]:
        assert d["lower"] <= d["predicted_sales"] <= d["upper"]


def test_forecasts_single_product(client):
    r = client.get("/api/v1/forecasts", params={"days": 3, "product_id": 11}).json()
    assert len(r["daily"]) == 3
    assert len(r["history"]) == 28
    assert client.get("/api/v1/forecasts", params={"product_id": 999}).status_code == 404


def test_forecasts_are_deterministic(client):
    a = client.get("/api/v1/forecasts").json()["summary"]
    b = client.get("/api/v1/forecasts").json()["summary"]
    assert a == b


def test_accuracy_and_backtest(client):
    m = client.get("/api/v1/forecasts/accuracy").json()
    assert m["sales"]["lstm"]["wape"] < m["sales"]["moyenne_4_semaines"]["wape"]
    bt = client.get("/api/v1/forecasts/backtest", params={"days": 30}).json()
    assert len(bt["daily"]) == 30


def test_losses(client):
    r = client.get("/api/v1/losses", params={"days": 30}).json()
    assert r["totals"]["waste"] > 0
    assert len(r["by_weekday"]) == 7
    assert len(r["daily"]) == 30


def test_comparison(client):
    r = client.get("/api/v1/comparison", params={"days": 30}).json()
    assert len(r["per_product"]) == 50
    assert len(r["tomorrow"]["recommendations"]) == 50
    assert "ai" in r["totals"] and "seat" in r["totals"]


def test_alerts_resolve_cycle(client):
    alerts = client.get("/api/v1/alerts").json()
    assert alerts, "au moins une alerte attendue"
    aid = alerts[0]["id"]
    assert client.post(f"/api/v1/alerts/{aid}/resolve", json={"note": "test"}).status_code == 200
    assert aid in {a["id"] for a in client.get("/api/v1/alerts", params={"status": "resolved"}).json()}
    assert client.post(f"/api/v1/alerts/{aid}/reopen").status_code == 200
    assert aid in {a["id"] for a in client.get("/api/v1/alerts").json()}
    assert client.post("/api/v1/alerts/inconnue/resolve").status_code == 404
    stats = client.get("/api/v1/alerts/stats").json()
    assert stats["active"] == len(alerts)


def test_products(client):
    r = client.get("/api/v1/products").json()
    assert len(r["products"]) == 50
    assert len(client.get("/api/v1/products", params={"category": "pain"}).json()["products"]) == 2
    d = client.get("/api/v1/products/11").json()
    assert d["product"]["name"] == "Croissant"
    assert len(d["forecast"]) == 7
    assert client.get("/api/v1/products/999").status_code == 404


def test_events(client):
    r = client.get("/api/v1/events").json()
    assert len(r["events"]) == 59


def test_chatbot(client):
    def ask(m):
        return client.post("/api/v1/chatbot/message", json={"message": m}).json()

    assert ask("Prévision des croissants pour demain")["intent"] == "forecast"
    assert "Croissant" in ask("Prévision des croissants pour demain")["response"]
    assert ask("Quelles alertes ?")["intent"] == "alerts"
    assert ask("gaspillage du mois")["intent"] == "waste"
    assert ask("précision du modèle")["intent"] == "accuracy"
    assert ask("blabla")["intent"] == "unknown"
