#TEST LAYOUT 2  hittas på: (http://localhost:8501)
import streamlit as st
import re
from numpy.random import default_rng as rng
import spacy #Ladda ner: py -m pip install spacy, py -m spacy download sv_core_news_sm
import sqlite3
import json
import datetime

# Hämta Resultat från databas
#Ändra path om den inte ligger i samma mapp
def safe_load_json(value):
    """Trygg JSON-laddning, även om fältet är null eller tomt."""
    try:
        if not value or value.strip() == "":
            return {}
        return json.loads(value)
    except json.JSONDecodeError:
        return {}

def load_latest_metrics_from_db(db_path="rag_results.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM results ORDER BY id DESC LIMIT 1")
    row = cursor.fetchone()
    #print("\n=== Kolumnvärden från senaste raden ===")
    #for i, val in enumerate(row):
    # print(f"{i}: {val[:200] if isinstance(val, str) else val}")  # visar bara 200 tecken
    conn.close()

    if not row:
        st.warning("Inga resultat hittades i databasen.")
        return None

    # Skapa samma struktur som tidigare hårdkodade 'metrics'
    metrics = {
        "compliance_report": json.loads(row[6] or "{}"),
        "ingestion": json.loads(row[7] or "{}"),
        "retrieval": json.loads(row[8] or "{}"),
        "generation": json.loads(row[9] or "{}"),
        "compliance": json.loads(row[12] or "{}"),  # metrics_compliance
        "metadata_compliance": json.loads(row[13] or "{}")
    }
    query = row[1]
    rag_answer = row[2]
    timestamp = row[14]

    # Extra: compliance_report finns i kolumn 6
    compliance_report = {}
    try:
        compliance_report = json.loads(row[6] or "{}")
    except Exception:
        pass

    return metrics, query, rag_answer, timestamp, compliance_report

# Ladda svenska modeller(känner igen ord med olika böjningar ex: "hatt" = "hatten")
nlp_sv = spacy.load("sv_core_news_sm")
st.set_page_config(layout="wide")

#Visar metric även om värdet är None
def safe_metric(label, value, suffix=""):
    if value is None:
        st.metric(label, "Ej tillgänglig")
    else:
        if isinstance(value, (int, float)):
            st.metric(label, f"{value}{suffix}")
        else:
            st.metric(label, str(value))

st.title("Utvärderingsresultat")

# Ny metrics värden:
metrics = load_latest_metrics_from_db("rag_results.db")
result = load_latest_metrics_from_db("rag_results.db")

if result:
    metrics, query, rag_answer, timestamp, compliance_report = result
else:
    st.stop()

#nyckelord för sökning
stopwords = {"hur", "på", "en", "och", "att", "som", "den", "det", "i", "ett", "and", "the"}
words = re.findall(r"\b[a-zåäöA-ZÅÄÖ]+\b", query.lower())

doc_q = nlp_sv(query)
# filtrera bort stopwords, icke-alfabetiska tokens och väldigt korta ord
keywords = [
    token.lemma_.lower()
    for token in doc_q
    if token.is_alpha and token.text.lower() not in stopwords and len(token.text) > 2]
#Markerar nyckelord  från fråga som matchar frågan och svaret med grön färg.
def highlight_keywords(text, keywords):
    doc = nlp_sv(text)
    highlighted_text = ""
    for token in doc:
        lemma = token.lemma_.lower()
        if lemma in keywords:
            highlighted_text += f"<span style='color:green; font-weight:bold'>{token.text}</span> "
        else:
            highlighted_text += token.text + " "
    return highlighted_text.strip()

verdict = compliance_report.get("verdict", "Ej tillgänglig")
compliance_score = metrics['compliance'].get("compliance score", 0.0)
summary = compliance_report.get("summary", "Ingen sammanfattning tillgänglig")
root_causes = compliance_report.get("root_causes", [])

#delar upp allt i olika columner så att de hamnar bredvid varandra!
col1, col2 = st.columns([2, 4])
# KOLUMN 1: LLM inforamtion + fråga
with col1:
    with st.container(border=True):
        st.subheader("💬 Fråga:")
        st.write(query)

        st.subheader("📋 Svar från RAG-systemet:")
        st.markdown(highlight_keywords(rag_answer, keywords), unsafe_allow_html=True)

# KOLUMN 2: Bedömning från compilance checker
with col2:
    with st.container(border=True):
        st.subheader("📊 Bedömning")
        # Omvandla compliance score till procent
        compliance_pct = round(compliance_score* 100, 1)
        st.metric("Compliance Score", f"{compliance_pct}%")
        # Bestäm färg baserat på nivå
        if compliance_pct >= 80:
            color = "#4CAF50"  # grönt
        elif compliance_pct >= 50:
            color = "#FFC107"  # gult
        else:
            color = "#F44336"  # rött
        # Skapa en HTML-stapel som progressbar
        st.markdown(f"""
        <div style="background-color:#e0e0e0; border-radius:10px; padding:3px; width:100%; height:30px;">
            <div style="width:{compliance_pct}%; background-color:{color}; height:100%; border-radius:10px; text-align:center; color:white; font-weight:bold;">
            </div>
        </div>
        """, unsafe_allow_html=True)

        avg_conf = metrics["compliance"].get("verification", {}).get("avg_confidence")
        st.write(f"**Verdict:** {verdict}")
        #confidence över vad?
        st.write(f"**Confidence:** {avg_conf *100:.0f}%")
        st.write(f"**Sammanfattning:** {summary}")
        timestamp = "2025-10-27T02:49:36.964478"
        date_only = datetime.datetime.fromisoformat(timestamp).date()
        st.write(f"Datum: {date_only}")


    with st.expander("Visa mer information"):
            # Root causes
            st.subheader("Detaljerad analys")
            st.markdown("**Identifierade orsaker:**")
            for cause in root_causes:
                st.markdown(f"- {cause}")

            st.divider()

            # Metriker INGESTION
            st.divider()
            st.subheader("Metriker:")
            st.markdown("Från ingestion:")
            colA, colB, colC = st.columns(3)
            with colA:
                safe_metric("Antal chunks", metrics["ingestion"].get("num_chunks"))
                safe_metric("Parsing success rate", f"{metrics['ingestion'].get('parsing_success_rate', 0)*100:.0f}%")
                safe_metric("Embedding trohet", f"{metrics['ingestion'].get('embedding_fidelity', 0)*100:.0f}%")

            with colB:
                safe_metric("Antal sidor", metrics["ingestion"].get("num_pages"))
                safe_metric("Totala antalet chars", metrics["ingestion"].get("total_chars"))

            with colC:
                safe_metric("Fullständighet metadata", f"{metrics['ingestion'].get('metadata_completeness',0)*100:.0f}%")
                safe_metric("Chunk coverage", f"{metrics['ingestion'].get('chunk_coverage_pct',0):.0f}%")

            doc_id = metrics["ingestion"].get("doc_id", "Ej tillgänglig")
            st.markdown(f"Dokument ID: <span style='font-family:monospace; font-size:14px; color:darkgrey;'>{doc_id}</span>", unsafe_allow_html=True)
            st.divider()

            #Metrik RETRIEVAL
            st.markdown("Från retriver:")
            colD, colE, colF = st.columns(3)
            with colD:
                st.metric("Lexical overlap", f"{metrics['retrieval']['lexical_overlap']*100:.1f}%")
                st.metric("Spridning mellan topträffar:", f"{metrics['retrieval']['topk_gap']:.1f}")
                safe_metric("Recall vs Gold", metrics["retrieval"]["recall_vs_gold"])    
            with colE:
                st.metric("Antalet träffar:", f"{metrics['retrieval']['num_hits']}")
                st.metric("Total tid:", f"{metrics['retrieval']['retrieval_latency']:.2f}s")
            with colF:
                st.metric("Förändring efter omrankning:", f"{metrics['retrieval']['re_rank_delta']:.2f}")
                st.metric("Antal källor:", f"{metrics['retrieval']['distinct_source_count']}")
            st.divider()

            #Metrik GENERATION
            st.markdown("Från generering:")
            colG, colH, colI = st.columns(3)
            with colG:
                st.metric("Svarslängd:", f"{metrics['generation']['answer_length']}")
                gen_model = metrics["compliance_report"]["models"]["rag_models"]["generation_model"]
                st.markdown(f"Genererings modell: <span style='font-family:monospace; font-size:14px; color:darkgrey;'>{gen_model}</span>", unsafe_allow_html=True)
            with colI:
                citations = metrics["generation"]["generator_declared_citations"]
                st.metric("Antal citat", len(citations))
            with st.expander("Visa citerade källor"):
                    for c in citations:
                        st.write(c)
            # Token logprob
            if metrics['generation']["token_logprob_stats"]["avg"] is not None:
                st.metric("Säkerhetsnivå i genererat svar:", f"{metrics['generation']['token_logprob_stats']['avg']:.2f}")
            else:
                st.write("Säkerhetsnivå i genererat svar: Ej tillgänglig")
            if metrics['generation']["preliminary_hallucinations_warnings"]:
                with st.expander("Hallucinationsvarningar"):
                    for warn in metrics['generation']["preliminary_hallucinations_warnings"]:
                        st.warning(warn) 
            st.divider()
            st.markdown("Från Compliance-utvärdering:")

            # COMPLIANCE värden
            colJ, colK, colL = st.columns(3)
            with colJ:
                safe_metric("Antal claims (extraherade)", metrics["compliance"]["claim_extraction"]["claim_count"])
                safe_metric("Totalt antal claims", metrics["compliance"]["num_claims"])
                emb_model = metrics["compliance_report"]["models"]["rag_models"]["embedding_model"]
                st.markdown(f"Embedding modell: <span style='font-family:monospace; font-size:14px; color:darkgrey;'>{emb_model}</span>", unsafe_allow_html=True)
                # lägg till claim text
            with colK:
                safe_metric("Verifierade claims", metrics["compliance"]["num_verified_claims"])
                safe_metric("Entailment ratio", f"{metrics['compliance']['verification']['entailment_ratio']*100:.1f}%")
            with colL:
                safe_metric("Contradiction ratio", f"{metrics['compliance']['verification']['contradiction_ratio']*100:.1f}%")
                safe_metric("Genomsnittligt förtroende", f"{metrics['compliance']['verification']['avg_confidence']*100:.1f}%")
            st.divider()

            st.subheader("Granskade påståenden")
            claims_list = metrics["metadata_compliance"]["claims"]

            colM, colN = st.columns(2)
            with colM:
                st.markdown("**Extraherade påståenden:**")
                if claims_list:
                    for c in claims_list:
                        st.markdown(f"- {c}")
                else:
                    st.info("Inga extraherade påståenden tillgängliga.")
            with colN:
                st.markdown("**Verifierade påståenden:**")
                st.metric("Genomsnittlig matchning i procent:", f"{metrics["compliance"]["claim_extraction"]["avg_match_score"]*100:.1f}%")
                
            # Top-k retrieval scores
            st.divider()
            st.markdown("**Relevanspoäng för toppträffar:**")
            st.bar_chart(metrics["retrieval"]["topk_scores"]) 
