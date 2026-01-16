import numpy as np
from sentence_transformers import SentenceTransformer, util
import pathlib
from datetime import datetime
import json
import time
import requests
import PyPDF2
import sys

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "gpt-oss:20b"

filepath = pathlib.Path('FORMAS_DE_PAGO.pdf')

if not filepath.exists():
    print(f"Error: No se encontró el archivo '{filepath}'.")
    exit()

class OutputLogger:
    def __init__(self, filename=None):
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"resultados_genetico_{timestamp}.txt"
        
        self.filename = filename
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')
        
        print(f"[LOG] Guardando resultados en: {filename}\n")
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()
        sys.stdout = self.terminal
        print(f"\n[OK] Resultados guardados exitosamente en: {self.filename}")

logger = OutputLogger()
sys.stdout = logger

def llamar_ollama(prompt, temperature=0.7, max_tokens=2000, timeout=300):

    url = f"{OLLAMA_BASE_URL}/api/generate"
    
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
            "num_ctx": 8192,
            "top_k": 40,
            "top_p": 0.9,
            "repeat_penalty": 1.1
        }
    }
    
    print(f"   → Enviando solicitud a Ollama (timeout: {timeout}s)...")
    inicio = time.time()
    
    try:
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        
        duracion = time.time() - inicio
        result = response.json()
        
        print(f"   ✓ Respuesta recibida en {duracion:.1f}s")
        
        thinking = result.get('thinking', '').strip()
        response_text = result.get('response', '').strip()
        
        if not response_text and thinking:
            if "answer:" in thinking.lower():
                parts = thinking.lower().split("answer:")
                if len(parts) > 1:
                    answer_part = thinking[thinking.lower().rfind("answer:") + 7:].strip()
                    answer_part = answer_part.strip('"\'')
                    if answer_part:
                        response_text = answer_part
            
            if not response_text:
                lines = thinking.strip().split('\n')
                for line in reversed(lines):
                    line = line.strip()
                    if line and not line.startswith('User') and not line.startswith('They'):
                        response_text = line.strip('"\'')
                        break
        
        if not response_text:
            print(f"   ⚠ Advertencia: No se pudo extraer respuesta")
            print(f"   Thinking: {thinking[:200]}")
            return ""
        
        print(f"   ✓ Respuesta extraída: {response_text[:100]}...")
        return response_text
        
    except requests.exceptions.Timeout:
        duracion = time.time() - inicio
        print(f"   ✗ TIMEOUT después de {duracion:.1f}s")
        return None
        
    except requests.exceptions.RequestException as e:
        print(f"   ✗ Error en llamada a Ollama: {e}")
        return None

def verificar_ollama():
    print("\n" + "="*120)
    print("VERIFICANDO CONEXION CON OLLAMA")
    print("="*120)
    
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        response.raise_for_status()
        modelos = response.json()
        print(f"✓ Ollama está corriendo en {OLLAMA_BASE_URL}")
        
        modelos_disponibles = [m['name'] for m in modelos.get('models', [])]
        print(f"✓ Modelos disponibles: {modelos_disponibles}")
        
        if OLLAMA_MODEL not in modelos_disponibles:
            print(f"\n⚠ ADVERTENCIA: El modelo '{OLLAMA_MODEL}' no está en la lista.")
            print(f"   Ejecuta: ollama pull {OLLAMA_MODEL}")
            respuesta = input("   ¿Deseas continuar de todos modos? (s/n): ")
            if respuesta.lower() != 's':
                exit()
        else:
            print(f"✓ Modelo '{OLLAMA_MODEL}' está disponible")
        
        print(f"\n⏳ Haciendo prueba rápida con el modelo...")
        test_response = llamar_ollama("Di solo 'OK'", temperature=0, max_tokens=10)
        if test_response and len(test_response) > 0:
            print(f"✓ Prueba exitosa. Respuesta: '{test_response[:50]}'")
            print(f"✓ Ollama está funcionando correctamente")
            return True
        else:
            print("✗ La prueba falló - respuesta vacía o None")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"✗ ERROR: No se puede conectar a Ollama en {OLLAMA_BASE_URL}")
        print("   Asegúrate de que Ollama esté corriendo:")
        print("   1. Abre una terminal")
        print("   2. Ejecuta: ollama serve")
        return False
    except Exception as e:
        print(f"✗ Error verificando Ollama: {e}")
        return False

embedder = SentenceTransformer('all-MiniLM-L6-v2')

def extraer_texto_pdf(filepath):
    try:
        with open(filepath, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            texto_completo = ""
            for page in pdf_reader.pages:
                texto_completo += page.extract_text() + "\n"
        return texto_completo
    except Exception as e:
        print(f"Error extrayendo texto del PDF: {e}")
        return ""

pdf_content = extraer_texto_pdf(filepath)
print(f"✓ PDF cargado: {len(pdf_content)} caracteres extraídos")

def llamar_ollama_con_contexto(system_prompt, user_prompt, temperature=0.0):
    """Llama a Ollama con un contexto limitado del PDF"""
    contexto_limitado = pdf_content[:4500]
    
    prompt_completo = f"""Contexto del documento (resumen):
{contexto_limitado}

{system_prompt}

{user_prompt}

Responde de forma concisa y directa."""
    
    return llamar_ollama(prompt_completo, temperature=temperature, max_tokens=500, timeout=300)

if not verificar_ollama():
    print("\n✗ No se puede continuar sin una conexión válida a Ollama.")
    exit()

class RateLimiter:
    def __init__(self, min_delay=0.5):
        self.min_delay = min_delay
        self.last_request_time = 0
    
    def wait(self):
        tiempo_transcurrido = time.time() - self.last_request_time
        tiempo_espera = self.min_delay - tiempo_transcurrido
        
        if tiempo_espera > 0:
            time.sleep(tiempo_espera)
        
        self.last_request_time = time.time()

rate_limiter = RateLimiter(min_delay=0.5)

def generar_prompts_dinamicos():
    print("\n" + "="*120)
    print("GENERANDO PROMPTS DINAMICOS CON OLLAMA")
    print("="*120)
    
    prompt_generacion = """Lista 10 roles diferentes para un asistente universitario que responde sobre formas de pago.

Ejemplos: "Eres un asistente formal", "Eres un amigo que ayuda", etc.

Responde SOLO con este JSON (sin markdown):
{"prompts": ["rol1", "rol2", "rol3", "rol4", "rol5", "rol6", "rol7", "rol8", "rol9", "rol10"]}"""
    
    try:
        rate_limiter.wait()
        
        respuesta_texto = llamar_ollama(prompt_generacion, temperature=0.7, max_tokens=1000, timeout=180)
        
        if not respuesta_texto:
            print("\n[!] Usando prompts predefinidos por fallo en generacion...")
            return [
                "Eres un asistente universitario formal y profesional.",
                "Eres un compañero estudiante amigable que ayuda.",
                "Eres un experto en administración educativa.",
                "Eres un consejero paciente y explicativo.",
                "Eres un tutor que da respuestas concisas.",
                "Eres un guía que proporciona información detallada.",
                "Eres un asistente directo y eficiente.",
                "Eres un orientador que motiva a los estudiantes.",
                "Eres un especialista en finanzas universitarias.",
                "Eres un ayudante que simplifica información compleja."
            ]
        
        respuesta_texto = respuesta_texto.strip()
        for delim in ['```json', '```', '`']:
            respuesta_texto = respuesta_texto.replace(delim, '')
        respuesta_texto = respuesta_texto.strip()
        
        datos = json.loads(respuesta_texto)
        prompts = datos['prompts']
        
        print(f"\n[+] Se generaron {len(prompts)} prompts exitosamente")
        for i, prompt in enumerate(prompts, 1):
            print(f"   {i}. {prompt[:80]}...")
        
        return prompts
    
    except Exception as e:
        print(f"[!] Error: {e}")
        print("Usando prompts predefinidos...")
        return [
            "Eres un asistente universitario formal y profesional.",
            "Eres un compañero estudiante amigable que ayuda.",
            "Eres un experto en administración educativa.",
            "Eres un consejero paciente y explicativo.",
            "Eres un tutor que da respuestas concisas.",
            "Eres un guía que proporciona información detallada.",
            "Eres un asistente directo y eficiente.",
            "Eres un orientador que motiva a los estudiantes.",
            "Eres un especialista en finanzas universitarias.",
            "Eres un ayudante que simplifica información compleja."
        ]

def generar_test_cases_dinamicos():
    print("\n" + "="*120)
    print("GENERANDO TEST CASES DINAMICOS CON OLLAMA (BASADOS EN EL PDF)")
    print("="*120)
    
    contexto_pdf = pdf_content[:4500] 
    
    prompt_generacion = f"""Lee el siguiente documento sobre formas de pago universitarias y genera 5 preguntas frecuentes con sus respuestas.

DOCUMENTO:
{contexto_pdf}

Genera 5 preguntas REALES (no pongas "pregunta1", "pregunta2", sino preguntas verdaderas como "¿Qué métodos de pago aceptan?") que un estudiante haría sobre este documento, con sus respuestas extraídas directamente del texto.

IMPORTANTE: Las preguntas deben ser preguntas completas y naturales, no etiquetas.

Ejemplo de formato correcto:
{{
  "casos": {{
    "¿Qué tarjetas aceptan?": "Se aceptan tarjetas VISA y MASTERCARD",
    "¿Cuál es el horario?": "De lunes a viernes de 8:00 a 18:00"
  }}
}}

Ahora genera TU JSON con 5 casos (sin markdown, sin explicaciones, solo el JSON):"""
    
    try:
        rate_limiter.wait()
        
        respuesta_texto = llamar_ollama(prompt_generacion, temperature=0.7, max_tokens=2500, timeout=300)
        
        if not respuesta_texto:
            print("\n[!] Usando test cases predefinidos por fallo en generacion...")
            return {
                "¿Qué métodos de pago aceptan?": "Aceptamos efectivo, tarjetas de débito y crédito, y transferencias bancarias.",
                "¿Cuál es el horario de caja?": "La caja atiende de lunes a viernes de 8:00 AM a 4:00 PM.",
                "¿Puedo pagar en línea?": "Sí, puedes realizar pagos en línea a través del portal estudiantil con tarjeta.",
                "¿Hay descuentos por pronto pago?": "Sí, ofrecemos 5% de descuento si pagas antes de la fecha límite.",
                "¿Aceptan pagos en cuotas?": "Sí, puedes dividir el pago en hasta 3 cuotas sin intereses."
            }
        
        respuesta_texto = respuesta_texto.strip()
        for delim in ['```json', '```', '`']:
            respuesta_texto = respuesta_texto.replace(delim, '')
        respuesta_texto = respuesta_texto.strip()
        
        try:
            datos = json.loads(respuesta_texto)
            test_cases = datos.get('casos', datos)
        except json.JSONDecodeError:
            print(f"[!] Error parseando JSON. Respuesta: {respuesta_texto[:200]}")
            print("Usando test cases predefinidos...")
            return {
                "¿Qué métodos de pago aceptan?": "Aceptamos efectivo, tarjetas de débito y crédito, y transferencias bancarias.",
                "¿Cuál es el horario de caja?": "La caja atiende de lunes a viernes de 8:00 AM a 4:00 PM.",
                "¿Puedo pagar en línea?": "Sí, puedes realizar pagos en línea a través del portal estudiantil con tarjeta.",
                "¿Hay descuentos por pronto pago?": "Sí, ofrecemos 5% de descuento si pagas antes de la fecha límite.",
                "¿Aceptan pagos en cuotas?": "Sí, puedes dividir el pago en hasta 3 cuotas sin intereses."
            }
        
        preguntas_validas = {}
        for pregunta, respuesta in test_cases.items():
            if not pregunta.lower().startswith('pregunta') and '?' in pregunta:
                preguntas_validas[pregunta] = respuesta
        
        if len(preguntas_validas) < 3:
            print(f"[!] Solo se generaron {len(preguntas_validas)} preguntas validas. Usando predefinidas...")
            return {
                "¿Qué métodos de pago aceptan?": "Aceptamos efectivo, tarjetas de débito y crédito, y transferencias bancarias.",
                "¿Cuál es el horario de caja?": "La caja atiende de lunes a viernes de 8:00 AM a 4:00 PM.",
                "¿Puedo pagar en línea?": "Sí, puedes realizar pagos en línea a través del portal estudiantil con tarjeta.",
                "¿Hay descuentos por pronto pago?": "Sí, ofrecemos 5% de descuento si pagas antes de la fecha límite.",
                "¿Aceptan pagos en cuotas?": "Sí, puedes dividir el pago en hasta 3 cuotas sin intereses."
            }
        
        if len(preguntas_validas) > 5:
            preguntas_validas = dict(list(preguntas_validas.items())[:5])
        
        print(f"\n[+] Se generaron {len(preguntas_validas)} casos de prueba exitosamente")
        print("\n" + "="*120)
        print("PREGUNTAS Y RESPUESTAS ESPERADAS GENERADAS:")
        print("="*120)
        for i, (pregunta, respuesta) in enumerate(preguntas_validas.items(), 1):
            print(f"\n[CASO {i}]")
            print(f"   PREGUNTA: {pregunta}")
            print(f"   RESPUESTA ESPERADA:")
            print(f"   {respuesta}")
            print(f"   {'-'*116}")
        print("="*120)
        
        return preguntas_validas
    
    except Exception as e:
        print(f"[!] Error: {e}")
        print("Usando test cases predefinidos...")
        return {
            "¿Qué métodos de pago aceptan?": "Aceptamos efectivo, tarjetas de débito y crédito, y transferencias bancarias.",
            "¿Cuál es el horario de caja?": "La caja atiende de lunes a viernes de 8:00 AM a 4:00 PM.",
            "¿Puedo pagar en línea?": "Sí, puedes realizar pagos en línea a través del portal estudiantil con tarjeta.",
            "¿Hay descuentos por pronto pago?": "Sí, ofrecemos 5% de descuento si pagas antes de la fecha límite.",
            "¿Aceptan pagos en cuotas?": "Sí, puedes dividir el pago en hasta 3 cuotas sin intereses."
        }

def generar_compatibilidad(num_prompts):
    print("\n" + "="*120)
    print("GENERANDO MATRIZ DE COMPATIBILIDAD")
    print("="*120)
    
    compatibilidad = {}
    
    for i in range(num_prompts):
        prompts_compatibles = [i]
        num_compatibles = np.random.randint(2, 4)
        otros_prompts = [j for j in range(num_prompts) if j != i]
        compatibles_aleatorios = np.random.choice(otros_prompts, min(num_compatibles, len(otros_prompts)), replace=False).tolist()
        
        prompts_compatibles.extend(compatibles_aleatorios)
        compatibilidad[i] = sorted(prompts_compatibles)
    
    print(f"\n[+] Matriz de compatibilidad generada para {num_prompts} prompts")
    for idx, compatibles in compatibilidad.items():
        print(f"   Prompt {idx}: compatible con {compatibles}")
    
    return compatibilidad

PROMPT_OPTIONS = generar_prompts_dinamicos()
TEST_CASES = generar_test_cases_dinamicos()

NUM_PROMPTS = len(PROMPT_OPTIONS)
NUM_GENES = 5 
PROMPT_COMPATIBILITY = generar_compatibilidad(NUM_PROMPTS)

def calcular_fitness_para_pregunta(individuo, pregunta, respuesta_esperada):
    prompt_indices = individuo[:]
    
    combined_prompt = "\n".join([
        f"Instruccion {i+1}: {PROMPT_OPTIONS[idx]}"
        for i, idx in enumerate(prompt_indices)
    ])
    
    print(f"\n   Evaluando Combinacion: {prompt_indices}")
    
    rate_limiter.wait()
    
    user_prompt = f"{combined_prompt}\n\nPregunta del estudiante: {pregunta}"
    
    try:
        generated_answer = llamar_ollama_con_contexto("", user_prompt, temperature=0.0)
        
        if not generated_answer:
            print(f"      Error: No se obtuvo respuesta de Ollama")
            return 0.0
            
    except Exception as e:
        print(f"      Error en Ollama: {e}")
        return 0.0

    embeddings = embedder.encode([respuesta_esperada, generated_answer], convert_to_tensor=True)
    similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
    
    print(f"      Similitud: {similarity:.4f}")
    
    return similarity

def cruzar_individuos(padre1, padre2, prob_cruce=0.9):
    if np.random.random() > prob_cruce:
        return [int(x) for x in padre1], [int(x) for x in padre2]
    
    punto_cruce = np.random.randint(1, len(padre1))
    
    hijo1 = [int(x) for x in (padre1[:punto_cruce] + padre2[punto_cruce:])]
    hijo2 = [int(x) for x in (padre2[:punto_cruce] + padre1[punto_cruce:])]
    
    print(f"\nCRUCE REALIZADO")
    print(f"   Punto de cruce: {punto_cruce}")
    print(f"   Padre 1: {padre1}")
    print(f"   Padre 2: {padre2}")
    print(f"   Hijo 1:  {hijo1}")
    print(f"   Hijo 2:  {hijo2}\n")
    
    return hijo1, hijo2

def mutar_individuo(individuo, prob_mutacion=0.60):
    mutado = individuo[:]
    genes_mutados = []
    
    for gene_idx in range(len(mutado)):
        if np.random.random() < prob_mutacion:
            current_prompt_idx = mutado[gene_idx]
            
            compatible_prompts = PROMPT_COMPATIBILITY.get(current_prompt_idx, list(range(NUM_PROMPTS)))
            
            nuevo_prompt = np.random.choice(compatible_prompts)
            mutado[gene_idx] = nuevo_prompt
            genes_mutados.append((gene_idx, current_prompt_idx, nuevo_prompt))
    
    if genes_mutados:
        print(f"MUTACION INTELIGENTE REALIZADA")
        print(f"   Individuo Original: {individuo}")
        for gene_idx, old_val, new_val in genes_mutados:
            print(f"   Gen {gene_idx}: {old_val} -> {new_val} (compatible)")
        print(f"   Individuo Mutado:   {mutado}\n")
    
    return mutado

class AlgoritmoGeneticoPorPregunta:
    def __init__(self, poblacion_inicial, num_generaciones, pregunta, respuesta_esperada):
        self.poblacion = poblacion_inicial
        self.num_generaciones = num_generaciones
        self.pregunta = pregunta
        self.respuesta_esperada = respuesta_esperada
        self.historial_fitness = []
        self.mejor_individuo_global = None
        self.mejor_fitness_global = -1
    
    def evaluar_poblacion(self, generacion):
        fitness_scores = []
        
        for idx_ind, individuo in enumerate(self.poblacion):
            print(f"\n{'='*120}")
            print(f"Evaluando Individuo {idx_ind + 1}/{len(self.poblacion)}")
            print(f"{'='*120}")
            
            fitness = calcular_fitness_para_pregunta(individuo, self.pregunta, self.respuesta_esperada)
            fitness_scores.append(fitness)
        
        return fitness_scores
    
    def seleccionar_mejores(self, fitness_scores, num_padres=3):
        indices_ordenados = sorted(range(len(fitness_scores)), 
                                  key=lambda i: fitness_scores[i], 
                                  reverse=True)
        
        padres_indices = indices_ordenados[:num_padres]
        padres = [self.poblacion[i] for i in padres_indices]
        
        print(f"\nSELECCION DE PADRES")
        print(f"   Mejores indices: {padres_indices}")
        fitness_padres = [f"{fitness_scores[i]:.4f}" for i in padres_indices]
        print(f"   Fitness: {fitness_padres}\n")
        
        return padres
    
    def generar_nueva_poblacion(self, padres, mejor_individuo_gen, mejor_fitness_gen):
        nueva_poblacion = []
        
        if self.mejor_fitness_global < mejor_fitness_gen:
            self.mejor_individuo_global = mejor_individuo_gen[:]
            self.mejor_fitness_global = mejor_fitness_gen
            print(f"\n🏆 NUEVO RECORD GLOBAL: Fitness = {self.mejor_fitness_global:.4f}")
            print(f"   Individuo: {self.mejor_individuo_global}")
        
        nueva_poblacion.append(self.mejor_individuo_global[:])
        print(f"\n✓ ELITISMO: Mejor individuo global preservado (fitness: {self.mejor_fitness_global:.4f})")
        
        nueva_poblacion.extend(padres)
        
        while len(nueva_poblacion) < len(self.poblacion):
            padre1 = padres[np.random.randint(0, len(padres))]
            padre2 = padres[np.random.randint(0, len(padres))]
            
            hijo1, hijo2 = cruzar_individuos(padre1, padre2)
            
            hijo1_mutado = mutar_individuo(hijo1)
            hijo2_mutado = mutar_individuo(hijo2)
            
            nueva_poblacion.append(hijo1_mutado)
            if len(nueva_poblacion) < len(self.poblacion):
                nueva_poblacion.append(hijo2_mutado)
        
        self.poblacion = nueva_poblacion[:len(self.poblacion)]
    
    def ejecutar(self):
        for gen in range(self.num_generaciones):
            print(f"\n\n{'='*120}")
            print(f"GENERACION {gen + 1}/{self.num_generaciones}")
            print(f"{'='*120}\n")
            
            fitness_scores = self.evaluar_poblacion(gen)
            
            mejor_fitness = max(fitness_scores)
            promedio_fitness = np.mean(fitness_scores)
            peor_fitness = min(fitness_scores)
            
            print(f"\n{'='*120}")
            print(f"ESTADISTICAS GENERACION {gen + 1}")
            print(f"{'='*120}")
            print(f"   Mejor Fitness:   {mejor_fitness:.4f}")
            print(f"   Promedio:        {promedio_fitness:.4f}")
            print(f"   Peor:            {peor_fitness:.4f}")
            print(f"   Rango:           {mejor_fitness - peor_fitness:.4f}")
            
            mejor_idx = fitness_scores.index(mejor_fitness)
            print(f"   Mejor Combinacion: {self.poblacion[mejor_idx]}")
            print(f"{'='*120}\n")
            
            self.historial_fitness.append({
                'gen': gen,
                'mejor': mejor_fitness,
                'promedio': promedio_fitness,
                'peor': peor_fitness
            })
            
            if mejor_fitness > self.mejor_fitness_global:
                self.mejor_individuo_global = self.poblacion[mejor_idx][:]
                self.mejor_fitness_global = mejor_fitness
            
            if gen < self.num_generaciones - 1:
                padres = self.seleccionar_mejores(fitness_scores)
                self.generar_nueva_poblacion(padres, self.poblacion[mejor_idx], mejor_fitness)
        
        return self.mejor_individuo_global, self.mejor_fitness_global

print("\n" + "="*120)
print("CONFIGURACION DEL ALGORITMO GENETICO")
print("="*120)
print(f"   Prompts Disponibles: {NUM_PROMPTS}")
print(f"   Test Cases (Preguntas): {len(TEST_CASES)}")
print(f"   Genes por Individuo: {NUM_GENES}")
print(f"   Poblacion por Pregunta: 10")
print(f"   Generaciones por Pregunta: 10")
print(f"   Modo: SECUENCIAL (una pregunta a la vez)")
print("="*120 + "\n")

resultados_finales = {}

for idx_pregunta, (pregunta, respuesta_esperada) in enumerate(TEST_CASES.items(), 1):
    print("\n\n" + "#"*120)
    print(f"{'='*120}")
    print(f"PREGUNTA {idx_pregunta}/{len(TEST_CASES)}")
    print(f"{'='*120}")
    print(f"[?] PREGUNTA: {pregunta}")
    print(f"[>] RESPUESTA ESPERADA: {respuesta_esperada}")
    print(f"{'='*120}")
    print("#"*120 + "\n")
    
    poblacion_inicial = [
        [int(x) for x in np.random.randint(0, NUM_PROMPTS, NUM_GENES)] for _ in range(10)
    ]
    
    print("\nPOBLACION INICIAL GENERADA:")
    for i, ind in enumerate(poblacion_inicial, 1):
        print(f"   Individuo {i}: {ind}")
    print()
    
    ag = AlgoritmoGeneticoPorPregunta(poblacion_inicial, num_generaciones=10, 
                                      pregunta=pregunta, respuesta_esperada=respuesta_esperada)
    mejor_solucion, mejor_fitness = ag.ejecutar()
    
    resultados_finales[pregunta] = {
        'mejor_solucion': mejor_solucion,
        'mejor_fitness': mejor_fitness,
        'historial': ag.historial_fitness
    }
    
    print("\n" + "="*120)
    print(f"OPTIMIZACION FINALIZADA PARA PREGUNTA {idx_pregunta}")
    print("="*120)
    print(f"\nPregunta: {pregunta}")
    print(f"Mejor Fitness GLOBAL: {mejor_fitness:.4f}")
    print(f"Mejor Combinacion GLOBAL: {mejor_solucion}")
    print(f"\nPrompts Seleccionados:")
    for i, idx in enumerate(mejor_solucion, 1):
        print(f"   {i}. {PROMPT_OPTIONS[idx]}")
    
    mejor_gen = -1
    for gen_data in ag.historial_fitness:
        if gen_data['mejor'] == mejor_fitness:
            mejor_gen = gen_data['gen']
            break
    
    if mejor_gen >= 0:
        print(f"\n[*] Mejor solucion encontrada en: Generacion {mejor_gen + 1}")
    
    print("="*120)

print("\n\n" + "*"*120)
print("="*120)
print("RESUMEN FINAL - MEJORES COMBINACIONES POR PREGUNTA")
print("="*120)
print("*"*120 + "\n")

for idx, (pregunta, datos) in enumerate(resultados_finales.items(), 1):
    print(f"\n{'='*120}")
    print(f"PREGUNTA {idx}: {pregunta}")
    print(f"{'='*120}")
    print(f"Fitness: {datos['mejor_fitness']:.4f}")
    print(f"Indices: {datos['mejor_solucion']}")
    print(f"\n[>>] PROMPT COMBINADO:")
    print("-" * 120)
    for i, idx_prompt in enumerate(datos['mejor_solucion'], 1):
        print(f"Instruccion {i}: {PROMPT_OPTIONS[idx_prompt]}")
    print("-" * 120)
    
    print(f"\nESTADISTICAS DE EVOLUCION:")
    print(f"{'Gen':<5} {'Mejor':<10} {'Promedio':<10} {'Peor':<10}")
    print("-"*50)
    for stats in datos['historial']:
        gen = stats['gen']
        best = stats['mejor']
        avg = stats['promedio']
        worst = stats['peor']
        print(f"{gen:<5} {best:<10.4f} {avg:<10.4f} {worst:<10.4f}")

print("\n" + "="*120)
print("OPTIMIZACION COMPLETA FINALIZADA")
print("="*120 + "\n")

logger.close()