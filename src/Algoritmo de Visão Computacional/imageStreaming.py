#!/usr/bin/env python3
import sys
import requests
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path

PASTA_FOTOS = Path(r"C:\Users\andre\OneDrive\Ambiente de Trabalho\FEUP\1º Ano - Mestrado\Projeto Integrador - UFSC\Fotografias_Recolhidas")

IRIUN_INDEX = 1

ANDROID_IP    = "172.20.10.2"
ANDROID_PORTA = 8080

# ─────────────────────────────
def criar_pasta():
    PASTA_FOTOS.mkdir(parents=True, exist_ok=True)

def nome_ficheiro(dispositivo: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return PASTA_FOTOS / f"{dispositivo}_{ts}.jpg"

# ─────────────────────────────
# Guardar imagem (robusto)
def guardar_imagem(frame, dispositivo):
    try:
        destino = nome_ficheiro(dispositivo)

        sucesso, buffer = cv2.imencode(".jpg", frame)
        if not sucesso:
            print("❌ Falha ao codificar imagem.")
            return

        with open(destino, "wb") as f:
            f.write(buffer)

        print(f"✅ Foto guardada: {destino}")

    except Exception as e:
        print(f"❌ Erro ao guardar: {e}")

# ─────────────────────────────
# 📱 iPhone (Iriun)
def modo_iphone():
    print("\n📱 MODO iPHONE (Iriun)\n")
    print("Antes de continuar:")
    print("  • Abrir app Iriun no iPhone")
    print("  • Abrir Iriun Webcam no PC")
    print("  • Estar na mesma rede Wi-Fi\n")

    input("Pressiona ENTER para iniciar...")

    cap = cv2.VideoCapture(IRIUN_INDEX, cv2.CAP_DSHOW)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("❌ Não foi possível ligar ao Iriun.")
        return

    print("\nC → capturar | E → sair\n")

    while True:
        ret, frame = cap.read()

        if not ret or frame is None:
            print("❌ Falha ao ler frame.")
            break

        cv2.imshow("iPhone (Iriun)", frame)

        tecla = cv2.waitKey(1) & 0xFF

        if tecla in [ord('e'), ord('E')]:
            break

        elif tecla in [ord('c'), ord('C')]:
            print("📸 Captura!")
            ret2, frame_final = cap.read()

            if ret2 and frame_final is not None:
                guardar_imagem(frame_final.copy(), "iphone")

    cap.release()
    cv2.destroyAllWindows()

# ─────────────────────────────
# 📱 Android (IP Webcam)
def modo_android():
    print("\n📱 MODO ANDROID (IP Webcam)\n")
    print("Antes de continuar:")
    print("  • Abrir app IP Webcam")
    print("  • Carregar em 'Start Server'")
    print(f"  • Confirmar IP: {ANDROID_IP}:{ANDROID_PORTA}\n")

    input("Pressiona ENTER para iniciar...")

    url = f"http://{ANDROID_IP}:{ANDROID_PORTA}/video"
    cap = cv2.VideoCapture(url)

    if not cap.isOpened():
        print("⚠️ Stream falhou → tentar snapshot direto")

    print("\nC → capturar | E → sair\n")

    while True:
        ret, frame = cap.read()

        if not ret or frame is None:
            print("❌ Stream falhou.")
            break

        cv2.imshow("Android", frame)

        tecla = cv2.waitKey(1) & 0xFF

        if tecla in [ord('e'), ord('E')]:
            break

        elif tecla in [ord('c'), ord('C')]:
            print("📸 Captura!")
            ret2, frame_final = cap.read()

            if ret2 and frame_final is not None:
                guardar_imagem(frame_final.copy(), "android")

    cap.release()
    cv2.destroyAllWindows()

# ─────────────────────────────
# MENU
def menu():
    criar_pasta()

    while True:
        print("\n──────── MENU ────────")
        print("1 → iPhone")
        print("2 → Android")

        op = input("Opção: ").lower()

        if op == "1":
            modo_iphone()
        elif op == "2":
            modo_android()
        elif op == "q":
            break
        else:
            print("Opção inválida.")

# ─────────────────────────────
if __name__ == "__main__":
    menu()