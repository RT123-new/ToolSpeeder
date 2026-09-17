"""Generator for held-out confirmatory benchmark evaluation dataset v1.0 (120 tasks)."""

import json
import random
from pathlib import Path
from typing import cast

FAMILY_TOOLS = {
    "documents": [
        ("fetch_tax_invoice", "Fetch and parse customer tax invoice by document ID", "doc_id", "DOC-"),
        ("read_nda_agreement", "Retrieve signed non-disclosure agreement document", "doc_id", "NDA-"),
        ("get_compliance_policy", "Inspect regional regulatory compliance policy document", "policy_id", "POL-"),
        ("retrieve_audit_certificate", "Retrieve verified security audit certificate document", "cert_id", "CERT-"),
    ],
    "customer_account": [
        ("lookup_customer_profile", "Lookup customer biographical and contact profile", "customer_id", "CUST-"),
        ("fetch_account_status", "Retrieve customer account billing status and credit limit", "account_id", "ACC-"),
        ("query_kyc_record", "Query Know-Your-Customer identity verification record", "kyc_id", "KYC-"),
        (
            "get_customer_billing_record",
            "Retrieve customer recurring billing method and payment terms",
            "customer_id",
            "CUST-",
        ),
    ],
    "orders": [
        ("get_purchase_order", "Retrieve enterprise purchase order and invoice itemization", "po_id", "PO-"),
        ("track_subscription_order", "Query active SaaS subscription contract and renewal tier", "sub_id", "SUB-"),
        ("lookup_return_manifest", "Lookup RMA customer goods return tracking manifest", "rma_id", "RMA-"),
        ("query_order_fulfillment", "Query warehouse dispatch and delivery fulfillment route", "order_id", "ORD-"),
    ],
    "inventory": [
        ("check_warehouse_stock", "Query current warehouse physical on-hand stock levels", "sku", "SKU-"),
        ("query_sku_availability", "Check regional warehouse SKU reservation availability", "sku", "SKU-"),
        (
            "fetch_backorder_estimate",
            "Estimate supplier delivery replenishment date for backordered item",
            "item_id",
            "ITEM-",
        ),
        ("inspect_batch_lot", "Inspect pharmaceutical or manufacturing batch lot expiration", "lot_id", "LOT-"),
    ],
    "code_repository": [
        ("get_git_commit", "Fetch git commit metadata, commit message, and parent hashes", "commit_sha", "SHA-"),
        ("inspect_pr_diff", "Inspect pull request unified diff and changed hunk lines", "pr_number", "PR-"),
        ("view_repository_tree", "List git tree directory hierarchy at specific ref", "tree_path", "DIR-"),
        ("lookup_blame_hunk", "Retrieve author attribution blame hunk for specific source line", "file_path", "SRC-"),
    ],
    "database_records": [
        ("query_financial_ledger", "Query double-entry financial ledger journal entries", "ledger_id", "LED-"),
        ("fetch_audit_row", "Retrieve immutable system audit database log row", "row_id", "ROW-"),
        ("get_user_record", "Fetch raw user table database entity by internal UUID", "user_uuid", "USR-"),
        ("lookup_transaction_entry", "Lookup card processing settlement transaction record", "txn_id", "TXN-"),
    ],
    "search_retrieval": [
        ("search_knowledge_base", "Perform semantic search over enterprise knowledge base articles", "query", "Q-"),
        ("query_vector_index", "Query ANN vector embeddings index for similar document chunks", "vector_id", "VEC-"),
        (
            "retrieve_documentation",
            "Retrieve official product technical architecture documentation",
            "doc_slug",
            "SLUG-",
        ),
        ("search_faq_catalog", "Search verified customer support frequently asked questions", "search_term", "TERM-"),
    ],
    "logs_observability": [
        ("fetch_apm_traces", "Fetch distributed APM trace spans and RPC call latency waterfall", "trace_id", "TRC-"),
        ("query_system_error_logs", "Query host system error log entries and stacktraces", "service_name", "SVC-"),
        ("get_metric_alert", "Retrieve active Datadog/Prometheus metric threshold alert", "alert_id", "ALT-"),
        ("inspect_ingress_traffic", "Inspect edge gateway ingress HTTP request access log", "request_id", "REQ-"),
    ],
    "configuration": [
        (
            "get_feature_flag_config",
            "Retrieve LaunchDarkly feature flag rollout rules and targeting",
            "flag_key",
            "FLAG-",
        ),
        (
            "read_runtime_env",
            "Read isolated production runtime environment variables and endpoints",
            "env_name",
            "ENV-",
        ),
        ("inspect_iam_role", "Inspect AWS IAM role permissions boundary and policy attachments", "role_arn", "ROLE-"),
        (
            "fetch_tenant_settings",
            "Fetch multi-tenant organization preference parameters and limits",
            "tenant_id",
            "TNT-",
        ),
    ],
    "analytics_metrics": [
        ("query_mrr_revenue", "Query monthly recurring revenue aggregation and customer expansion", "period", "PER-"),
        ("get_user_retention_kpi", "Fetch cohort retention percentage curve and churn rate KPI", "cohort_id", "COH-"),
        ("calculate_churn_rate", "Calculate logo and net-dollar revenue churn rate for time window", "quarter", "QTR-"),
        (
            "lookup_dau_mau_ratio",
            "Lookup daily active users to monthly active users engagement ratio",
            "app_id",
            "APP-",
        ),
    ],
}

flat_tools = []
for fam, tlist in FAMILY_TOOLS.items():
    for name, desc, arg, pfx in tlist:
        flat_tools.append({"name": name, "description": desc, "arg_name": arg, "prefix": pfx, "family": fam})

FAMILY_PROMPTS = {
    "documents": {
        "fetch_tax_invoice": [
            ("Please fetch customer tax invoice {prefix}{num} for the annual fiscal audit.", "LOW"),
            ("Can you inspect document {prefix}{num} to confirm billing tax line items?", "MEDIUM"),
            ("Retrieve tax statement record for transaction {prefix}{num}.", "HIGH"),
        ],
        "read_nda_agreement": [
            ("Retrieve the countersigned non-disclosure agreement document {prefix}{num}.", "LOW"),
            ("Check legal terms in agreement file {prefix}{num}.", "MEDIUM"),
            ("Review non-disclosure agreement clauses in {prefix}{num}.", "HIGH"),
        ],
        "get_compliance_policy": [
            ("Inspect regional compliance policy {prefix}{num} for GDPR cross-border rules.", "LOW"),
            ("What are the corporate compliance guidelines in policy document {prefix}{num}?", "HIGH"),
        ],
        "retrieve_audit_certificate": [
            ("Fetch SOC-2 Type II audit certificate {prefix}{num} for client review.", "MEDIUM"),
            ("Verify security audit accreditation document {prefix}{num}.", "HIGH"),
        ],
        "no_spec": [
            "What is the corporate retention schedule for financial invoices?",
            "Can you explain the difference between our EU and US compliance policies?",
        ],
    },
    "customer_account": {
        "lookup_customer_profile": [
            ("Lookup contact details and address for customer profile {prefix}{num}.", "LOW"),
            ("Retrieve customer information for entity {prefix}{num}.", "MEDIUM"),
            ("Get full details for customer {prefix}{num}.", "HIGH"),
        ],
        "fetch_account_status": [
            ("Check whether customer account {prefix}{num} is active and has good standing.", "LOW"),
            ("Inspect billing account status for {prefix}{num}.", "MEDIUM"),
            ("Verify credit limit and payment standing for account {prefix}{num}.", "HIGH"),
        ],
        "query_kyc_record": [
            ("Retrieve KYC passport verification status for user verification {prefix}{num}.", "LOW"),
            ("Check identity compliance check for subject {prefix}{num}.", "MEDIUM"),
        ],
        "get_customer_billing_record": [
            ("Fetch recurring billing profile and credit card terms for customer {prefix}{num}.", "HIGH"),
            ("Review billing terms and payment schedule for customer {prefix}{num}.", "HIGH"),
        ],
        "no_spec": [
            "How do we handle VIP customer onboarding when documentation is incomplete?",
            "What is the standard SLA for reviewing escalated KYC alerts?",
        ],
    },
    "orders": {
        "get_purchase_order": [
            ("Retrieve purchase order {prefix}{num} including line items and unit prices.", "LOW"),
            ("Fetch purchase details for vendor order {prefix}{num}.", "MEDIUM"),
            ("Check procurement order {prefix}{num}.", "HIGH"),
        ],
        "track_subscription_order": [
            ("Look up the SaaS subscription contract tier for subscription {prefix}{num}.", "LOW"),
            ("Inspect recurring contract order {prefix}{num}.", "MEDIUM"),
            ("Check subscription terms for contract {prefix}{num}.", "HIGH"),
        ],
        "lookup_return_manifest": [
            ("Retrieve the return goods manifest for RMA {prefix}{num} at dock 3.", "LOW"),
            ("Check returned package status for RMA {prefix}{num}.", "MEDIUM"),
        ],
        "query_order_fulfillment": [
            ("Query warehouse dispatch routing and courier tracking for order {prefix}{num}.", "HIGH"),
            ("Check fulfillment progress and delivery dispatch for order {prefix}{num}.", "HIGH"),
        ],
        "no_spec": [
            "What is our return policy for opened hardware components?",
            "Can a customer modify their order after it enters fulfillment dispatch?",
        ],
    },
    "inventory": {
        "check_warehouse_stock": [
            ("Check on-hand warehouse physical inventory units for SKU {prefix}{num}.", "LOW"),
            ("Inspect inventory count for product {prefix}{num}.", "MEDIUM"),
            ("How many items do we have available for {prefix}{num}?", "HIGH"),
        ],
        "query_sku_availability": [
            ("Verify regional reservation availability for SKU {prefix}{num}.", "LOW"),
            ("Check regional reservation units for product {prefix}{num}.", "MEDIUM"),
            ("Can we allocate 500 units of {prefix}{num} to east-coast fulfillment?", "HIGH"),
        ],
        "fetch_backorder_estimate": [
            ("Fetch the estimated replenishment date for backordered item {prefix}{num}.", "LOW"),
            ("When is shipment expected for delayed item {prefix}{num}?", "MEDIUM"),
        ],
        "inspect_batch_lot": [
            ("Inspect expiration date and QA signoff for manufacturing batch lot {prefix}{num}.", "HIGH"),
            ("Check batch quality control report for production lot {prefix}{num}.", "HIGH"),
        ],
        "no_spec": [
            "What is the formula used for safety stock calculation across regional hubs?",
            "How do we handle write-offs for damaged goods during pallet transfers?",
        ],
    },
    "code_repository": {
        "get_git_commit": [
            ("Fetch git commit details and author metadata for commit {prefix}{num}.", "LOW"),
            ("Inspect changes committed in revision {prefix}{num}.", "MEDIUM"),
            ("Review commit details for ref {prefix}{num}.", "HIGH"),
        ],
        "inspect_pr_diff": [
            ("Inspect pull request unified diff and line hunks for PR #{prefix}{num}.", "LOW"),
            ("Check code modifications in pull request {prefix}{num}.", "MEDIUM"),
            ("Review proposed code edits in PR {prefix}{num}.", "HIGH"),
        ],
        "view_repository_tree": [
            ("List repository tree structure at directory {prefix}{num}.", "LOW"),
            ("Explore repository hierarchy under path {prefix}{num}.", "MEDIUM"),
        ],
        "lookup_blame_hunk": [
            ("Retrieve git blame attribution for line range in file {prefix}{num}.", "HIGH"),
            ("Check author attribution for lines in module {prefix}{num}.", "HIGH"),
        ],
        "no_spec": [
            "What is the merge policy for release branches during code freeze?",
            "Can you explain our branch protection rules for direct pushes to main?",
        ],
    },
    "database_records": {
        "query_financial_ledger": [
            ("Query financial ledger journal entries for ledger book {prefix}{num}.", "LOW"),
            ("Retrieve balance sheet journal entries for ID {prefix}{num}.", "MEDIUM"),
            ("Check debit and credit records for ledger {prefix}{num}.", "HIGH"),
        ],
        "fetch_audit_row": [
            ("Fetch immutable security audit row {prefix}{num} from database log.", "LOW"),
            ("Retrieve audit event log row {prefix}{num}.", "MEDIUM"),
            ("Check database transaction audit trace for record {prefix}{num}.", "HIGH"),
        ],
        "get_user_record": [
            ("Retrieve database row for user UUID {prefix}{num}.", "LOW"),
            ("Query raw user entity record for identifier {prefix}{num}.", "MEDIUM"),
        ],
        "lookup_transaction_entry": [
            ("Lookup payment gateway settlement record for transaction {prefix}{num}.", "HIGH"),
            ("Fetch financial settlement transaction {prefix}{num}.", "HIGH"),
        ],
        "no_spec": [
            "What database isolation level is required for cross-shard ledger transfers?",
            "How are soft-deleted user records handled in compliance with GDPR purge requests?",
        ],
    },
    "search_retrieval": {
        "search_knowledge_base": [
            ("Search knowledge base articles for query {prefix}{num}.", "LOW"),
            ("Find technical documentation matching topic {prefix}{num}.", "MEDIUM"),
            ("Search support articles regarding {prefix}{num}.", "HIGH"),
        ],
        "query_vector_index": [
            ("Query embeddings vector index for nearest semantic neighbors of {prefix}{num}.", "LOW"),
            ("Search nearest embeddings for document chunk {prefix}{num}.", "MEDIUM"),
            ("Perform semantic similarity lookup for query embedding {prefix}{num}.", "HIGH"),
        ],
        "retrieve_documentation": [
            ("Retrieve system architecture documentation for slug {prefix}{num}.", "LOW"),
            ("Fetch technical guide document {prefix}{num}.", "MEDIUM"),
        ],
        "search_faq_catalog": [
            ("Search customer FAQ catalog for user query {prefix}{num}.", "HIGH"),
            ("Find frequent support questions relating to {prefix}{num}.", "HIGH"),
        ],
        "no_spec": [
            "How does our vector index perform hybrid keyword and semantic scoring?",
            "What is the indexing frequency for updated knowledge base articles?",
        ],
    },
    "logs_observability": {
        "fetch_apm_traces": [
            ("Fetch APM trace waterfall spans for distributed trace {prefix}{num}.", "LOW"),
            ("Inspect RPC latency bottlenecks in trace {prefix}{num}.", "MEDIUM"),
            ("Analyze span timings for request trace {prefix}{num}.", "HIGH"),
        ],
        "query_system_error_logs": [
            ("Query error logs and stacktraces for backend microservice {prefix}{num}.", "LOW"),
            ("Check backend runtime exceptions in service {prefix}{num}.", "MEDIUM"),
            ("Inspect recent failure logs for service {prefix}{num}.", "HIGH"),
        ],
        "get_metric_alert": [
            ("Retrieve Datadog threshold alert details for firing alert {prefix}{num}.", "LOW"),
            ("Check alert status and firing condition for alert ID {prefix}{num}.", "MEDIUM"),
        ],
        "inspect_ingress_traffic": [
            ("Inspect edge gateway access logs for request UUID {prefix}{num}.", "HIGH"),
            ("Review HTTP gateway logs for incoming call {prefix}{num}.", "HIGH"),
        ],
        "no_spec": [
            "What is the retention period for raw high-cardinality APM trace spans?",
            "How should on-call engineers respond when P99 latency alerts trigger?",
        ],
    },
    "configuration": {
        "get_feature_flag_config": [
            ("Retrieve LaunchDarkly targeting rules and rollouts for flag {prefix}{num}.", "LOW"),
            ("Check rollout percentage for feature toggle {prefix}{num}.", "MEDIUM"),
            ("Inspect feature status for key {prefix}{num}.", "HIGH"),
        ],
        "read_runtime_env": [
            ("Read environment variable configuration for runtime deployment {prefix}{num}.", "LOW"),
            ("Check production deployment configuration for {prefix}{num}.", "MEDIUM"),
            ("Inspect server environment settings for {prefix}{num}.", "HIGH"),
        ],
        "inspect_iam_role": [
            ("Inspect IAM role permissions boundary and policies for role {prefix}{num}.", "LOW"),
            ("Check permission attachments on role {prefix}{num}.", "MEDIUM"),
        ],
        "fetch_tenant_settings": [
            ("Fetch multi-tenant configuration limits and quotas for tenant {prefix}{num}.", "HIGH"),
            ("Review tenant resource quotas and limits for tenant {prefix}{num}.", "HIGH"),
        ],
        "no_spec": [
            "What is the blast-radius policy for updating feature flags in production?",
            "How do we rotate IAM service account keys across multi-region clusters?",
        ],
    },
    "analytics_metrics": {
        "query_mrr_revenue": [
            ("Query monthly recurring revenue (MRR) expansion numbers for period {prefix}{num}.", "LOW"),
            ("Calculate recurring revenue numbers for fiscal period {prefix}{num}.", "MEDIUM"),
            ("Analyze revenue breakdown for period {prefix}{num}.", "HIGH"),
        ],
        "get_user_retention_kpi": [
            ("Fetch weekly cohort retention percentage curve for signup cohort {prefix}{num}.", "LOW"),
            ("Inspect cohort churn KPI for cohort {prefix}{num}.", "MEDIUM"),
            ("Review cohort retention KPI for cohort {prefix}{num}.", "HIGH"),
        ],
        "calculate_churn_rate": [
            ("Calculate logo churn and net dollar retention rate for quarter {prefix}{num}.", "LOW"),
            ("Compute customer churn metrics for quarter {prefix}{num}.", "MEDIUM"),
        ],
        "lookup_dau_mau_ratio": [
            ("Lookup daily to monthly active user ratio (DAU/MAU) for application {prefix}{num}.", "HIGH"),
            ("Retrieve active user engagement ratio for app {prefix}{num}.", "HIGH"),
        ],
        "no_spec": [
            "What statistical methodology is used to smooth DAU/MAU reporting across holidays?",
            "How is net dollar retention defined when customer expansion precedes upgrades?",
        ],
    },
}


def generate_dataset() -> list[dict]:
    rng = random.Random(42)
    tasks = []
    task_idx = 1
    k_options = [2, 4, 8, 16]
    k_idx = 0

    for fam, tdefs in FAMILY_TOOLS.items():
        fam_tools_map = {t[0]: t for t in tdefs}
        prompts_data = FAMILY_PROMPTS[fam]

        fam_tasks = []
        for tool_name, t_prompts in prompts_data.items():
            if tool_name == "no_spec":
                continue
            t_meta = fam_tools_map[tool_name]
            for prompt_tpl, amb in cast(list[tuple[str, str]], t_prompts):
                num = rng.randint(1000, 9999)
                prompt = prompt_tpl.format(prefix=t_meta[3], num=num)
                arg_val = f"{t_meta[3]}{num}"
                fam_tasks.append((tool_name, prompt, amb, {t_meta[2]: arg_val}))

        fam_tasks = fam_tasks[:10]

        for ns_prompt in cast(list[str], prompts_data["no_spec"]):
            fam_tasks.append(("no_speculation", ns_prompt, "LOW", {}))

        for target_tool, prompt, ambiguity, args in fam_tasks:
            k = k_options[k_idx % len(k_options)]
            k_idx += 1

            cand_list = []
            if target_tool != "no_speculation":
                cand_list.append(
                    {
                        "candidate_id": f"c1_{target_tool}",
                        "tool_name": target_tool,
                        "tool_family": fam,
                        "arguments": args,
                        "is_read_only": True,
                    }
                )

            distractors = []
            if ambiguity == "HIGH":
                same_fam = [t for t in flat_tools if t["family"] == fam and t["name"] != target_tool]
                distractors.extend(same_fam)
            elif ambiguity == "MEDIUM":
                same_fam = [t for t in flat_tools if t["family"] == fam and t["name"] != target_tool]
                diff_fam = [t for t in flat_tools if t["family"] != fam]
                rng.shuffle(diff_fam)
                distractors.extend(same_fam[:1] + diff_fam)
            else:
                diff_fam = [t for t in flat_tools if t["family"] != fam]
                rng.shuffle(diff_fam)
                distractors.extend(diff_fam)

            for d in distractors:
                if len(cand_list) >= k:
                    break
                cid = f"c{len(cand_list) + 1}_{d['name']}"
                cand_num = rng.randint(1000, 9999)
                cand_args = {d["arg_name"]: f"{d['prefix']}{cand_num}"}
                cand_list.append(
                    {
                        "candidate_id": cid,
                        "tool_name": d["name"],
                        "tool_family": d["family"],
                        "arguments": cand_args,
                        "is_read_only": True,
                    }
                )

            if len(cand_list) < k:
                for d in flat_tools:
                    if len(cand_list) >= k:
                        break
                    if any(c["tool_name"] == d["name"] for c in cand_list):
                        continue
                    cid = f"c{len(cand_list) + 1}_{d['name']}"
                    cand_num = rng.randint(1000, 9999)
                    cand_list.append(
                        {
                            "candidate_id": cid,
                            "tool_name": d["name"],
                            "tool_family": d["family"],
                            "arguments": {d["arg_name"]: f"{d['prefix']}{cand_num}"},
                            "is_read_only": True,
                        }
                    )

            rng.shuffle(cand_list)
            clean_cands = []
            for c_i, c in enumerate(cand_list):
                clean_cands.append(
                    {
                        "candidate_id": f"c{c_i + 1}",
                        "tool_name": c["tool_name"],
                        "tool_family": c["tool_family"],
                        "arguments": c["arguments"],
                        "is_read_only": c["is_read_only"],
                    }
                )

            task_obj = {
                "task_id": f"heldout_{task_idx:03d}",
                "family": fam,
                "prompt": prompt,
                "target_tool": target_tool,
                "ambiguity": ambiguity,
                "candidate_count": k,
                "candidates": clean_cands,
            }
            tasks.append(task_obj)
            task_idx += 1

    return tasks


if __name__ == "__main__":
    tasks = generate_dataset()
    out_p = Path("/Users/regtroka/Downloads/ToolSpeed/benchmarks/data/held_out_speculation_tasks_v1.0.json")
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(tasks, indent=2) + "\n")
    print(f"Successfully generated {len(tasks)} held-out tasks to {out_p}")
