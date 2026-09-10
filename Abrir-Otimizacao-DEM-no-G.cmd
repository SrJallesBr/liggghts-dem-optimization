@echo off
setlocal

rem Monta o Google Drive no WSL e executa o LIGGGHTS como o usuario normal.
set "ROOT=/mnt/g/My Drive/Programas/Liggghts/Cacterizacao/Repouso e Draftfown/otm_doe_drive_v9_6x2"

wsl.exe -d Ubuntu-24.04 -u root -- bash -lc "umount /mnt/g 2>/dev/null || true; mount -t drvfs G: /mnt/g && exec runuser -u henrique_jalles -- bash -lc 'cd \"%ROOT%\" && exec bash ./rodar_otimizacao_wsl.sh'"
echo.
echo Processo encerrado. Pressione qualquer tecla para fechar.
pause >nul

