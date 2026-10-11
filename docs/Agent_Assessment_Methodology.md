# Agent Assessment Methodology

This project operationalises a prompt-and-template interoperability assessment method as a staged AI-assisted workflow.

## Core idea

The tool is designed to support an interoperability assessor. It does not replace legal or institutional judgement. It helps the assessor identify the relevant obligations, interoperability effects, risks and implementation recommendations in a structured and repeatable way.

## Assessment stages

1. **Instrument profile**  
   Identifies the type of document, responsible body, jurisdiction, status, digital-public-service elements and cross-border relevance.

2. **Binding requirements extraction**  
   Extracts obligations, prohibitions, conditions, criteria and procedural duties that may affect digital public services or supporting information systems.

3. **Interoperable Europe Act trigger assessment**  
   Applies the trigger logic: whether the instrument defines or substantially modifies binding requirements affecting digital public services with cross-border or trans-European relevance.

4. **Service and stakeholder mapping**  
   Maps affected digital public services, stakeholders, actors, dependencies and cross-border or inter-authority data flows.

5. **Interoperability layer assessment**  
   Analyses effects across legal, organisational, semantic, technical, governance, user-experience and security/trust dimensions.

6. **Sensitive terms and drafting-pattern analysis**  
   Uses the internal lexicon as a signal generator. Detected terms are interpreted in context and may become legacy administrative patterns or implementation under-specification findings.

7. **Solution and reuse mapping**  
   Maps identified gaps to possible interoperability frameworks, standards, reusable solutions and implementation assets. EIRA is used here when selected as an architecture-oriented reference.

8. **Legal-to-technical translation**  
   Translates legal requirements into possible implementation needs while preserving the distinction between legal obligations and implementation recommendations.

9. **Risk prioritisation**  
   Consolidates overlapping issues into a canonical risk register used consistently across the report.

10. **Recommendations and roadmap**  
   Produces corrective measures, drafting recommendations, governance actions, technical specifications or operational guidance, linked where possible to risks and binding requirements.

11. **Validation and reporting**  
   Validates unsupported or overconfident claims and generates the selected report type.

## Assessment basis roles

- **Interoperable Europe Act**: legal trigger logic.
- **IE Act Guidelines**: assessment and reporting method.
- **EIF 3.0**: interoperability layers and public-sector interoperability principles.
- **EIRA 6.1.0**: architecture and reuse-oriented reasoning, when selected.
- **Uploaded guideline**: supplementary domain context when provided and relevant.

## Output modes

- **Risk Analysis Report**: shorter screening-style report focused on trigger posture, binding requirements, affected services, sensitive terms and main risks.
- **Full Assessment Report with Recommendations**: extended report including corrective measures, recommendations, reuse opportunities, legal-to-technical reasoning, roadmap, validation and traceability.

## Human validation

The generated report is a decision-support output. It should be reviewed against the final legal text, implementing measures, technical specifications and institutional evidence before being used for official decisions.
