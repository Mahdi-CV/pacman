import pygame
import random
import math
import os
import numpy as np
import wave
import heapq

SURVIVE_TO_WIN = 15   # seconds


def start_screen(screen, clock):
    """Display start screen and wait for ENTER key"""
    title_font = pygame.font.Font(None, 72)
    instruction_font = pygame.font.Font(None, 36)
    
    while True:
        screen.fill(BLACK)
        title = title_font.render("PAC-MAN", True, YELLOW)
        instruction = instruction_font.render("Press ENTER to Play", True, WHITE)
        
        # Draw title and instruction
        screen.blit(title, (WIDTH//2 - title.get_width()//2, HEIGHT//2 - 50))
        screen.blit(instruction, (WIDTH//2 - instruction.get_width()//2, HEIGHT//2 + 50))
        
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False  # Quit game
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    return True  # Start game
        
        clock.tick(FPS)

def bfs(maze, start, goal):
    from collections import deque
    queue = deque([start])
    came_from = {start: None}
    directions = [(0,1), (0,-1), (1,0), (-1,0)]

    while queue:
        current = queue.popleft()
        if current == goal:
            break
        for dx, dy in directions:
            next_node = (current[0] + dx, current[1] + dy)
            x, y = next_node
            if not (0 <= x < COLS and 0 <= y < ROWS):
                continue
            if maze[y][x] == '#' or next_node in came_from:
                continue
            queue.append(next_node)
            came_from[next_node] = current

    path = []
    current = goal
    while current != start:
        prev = came_from.get(current)
        if prev is None:
            return []
        path.append(current)
        current = prev
    path.reverse()
    return path

def greedy(maze, start, goal):
    queue = [(heuristic(start, goal), start)]
    came_from = {start: None}
    directions = [(0,1), (0,-1), (1,0), (-1,0)]

    while queue:
        _, current = heapq.heappop(queue)
        if current == goal:
            break
        for dx, dy in directions:
            next_node = (current[0] + dx, current[1] + dy)
            x, y = next_node
            if not (0 <= x < COLS and 0 <= y < ROWS):
                continue
            if maze[y][x] == '#' or next_node in came_from:
                continue
            heapq.heappush(queue, (heuristic(next_node, goal), next_node))
            came_from[next_node] = current

    path = []
    current = goal
    while current != start:
        prev = came_from.get(current)
        if prev is None:
            return []
        path.append(current)
        current = prev
    path.reverse()
    return path

def random_walk(maze, start, goal=None):
    directions = [(0,1), (0,-1), (1,0), (-1,0)]
    valid = [(start[0]+dx, start[1]+dy) for dx, dy in directions
             if 0 <= start[0]+dx < COLS and 0 <= start[1]+dy < ROWS and maze[start[1]+dy][start[0]+dx] != '#']
    random.shuffle(valid)
    return [valid[0]] if valid else []

def heuristic(a, b):
    # Manhattan distance
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar(maze, start, goal):
    queue = []
    heapq.heappush(queue, (0, start))
    came_from = {start: None}
    cost_so_far = {start: 0}
    directions = [(0,1), (0,-1), (1,0), (-1,0)]

    while queue:
        _, current = heapq.heappop(queue)

        if current == goal:
            break

        for dx, dy in directions:
            next_node = (current[0] + dx, current[1] + dy)
            x, y = next_node
            if not (0 <= x < COLS and 0 <= y < ROWS):
                continue
            if maze[y][x] == '#':
                continue

            new_cost = cost_so_far[current] + 1
            if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                cost_so_far[next_node] = new_cost
                priority = new_cost + heuristic(goal, next_node)
                heapq.heappush(queue, (priority, next_node))
                came_from[next_node] = current

    # Reconstruct path
    path = []
    current = goal
    while current != start:
        prev = came_from.get(current)
        if prev is None:
            return []  # No path
        path.append(current)
        current = prev
    path.reverse()
    return path


# Initialize Pygame
pygame.init()
pygame.mixer.init()
pygame.key.set_repeat(0) 

# Sound file paths
SOUND_DIR = "sounds"
SOUND_FILES = {
    'waka1': 'waka1.wav',
    'waka2': 'waka2.wav',
    'siren': 'siren.wav',
    # 'power_pellet': 'power_pellet.wav',
    # 'ghost_eaten': 'ghost_eaten.wav',
    'death': 'death.wav',
    'start': 'game_start.wav',
    'victory': 'victory.wav',
}

# Create sounds directory if not exists
if not os.path.exists(SOUND_DIR):
    os.makedirs(SOUND_DIR)

def generate_sound(frequency, duration=0.1, wave_type='square'):
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    if wave_type == 'square':
        wave_data = 0.5 * np.sign(np.sin(2 * np.pi * frequency * t))
    else:  # sine
        wave_data = 0.5 * np.sin(2 * np.pi * frequency * t)
    
    # Convert to 16-bit integers
    wave_data = np.int16(wave_data * 32767)
    
    # Convert to stereo (duplicate mono to both channels)
    stereo_wave = np.column_stack((wave_data, wave_data))
    return stereo_wave

# Generate and save missing sounds
for name, params in [
    ('waka1', (440, 0.08, 'square')),
    ('waka2', (660, 0.08, 'square')),
    # ('power_pellet', (1000, 0.8, 'sine')),
    # ('ghost_eaten', (1600, 0.3, 'square')),
    ('death', (200, 1.5, 'sine')),
    ('victory', (880, 2.0, 'sine'))
]:
    path = os.path.join(SOUND_DIR, SOUND_FILES[name])
    if not os.path.exists(path):
        wave_data = generate_sound(*params)
        
        # Save using wave module
        with wave.open(path, 'w') as wav_file:
            wav_file.setnchannels(2)       # Stereo
            wav_file.setsampwidth(2)       # 2 bytes per sample
            wav_file.setframerate(44100)
            wav_file.writeframes(wave_data.tobytes())


# Load sounds
# Load sounds
sounds = {
    name: pygame.mixer.Sound(os.path.join(SOUND_DIR, fname))
    for name, fname in SOUND_FILES.items()
    if os.path.exists(os.path.join(SOUND_DIR, fname))  # Avoid loading missing files
}




# Game constants
CELL_SIZE = 20
COLS = 28
ROWS = 31
WIDTH = COLS * CELL_SIZE
HEIGHT = ROWS * CELL_SIZE + 80  # Extra space for UI
FPS          = 10          # instead of 10
PAC_SPEED    = 3           # pixels per frame  (60 FPS × 4 px ≈ 240 px/s)
GHOST_SPEED  = 0.8 

# Colors
BLACK = (0, 0, 0)
# BLACK = (88, 90, 93)
# YELLOW = (207, 168, 61)
YELLOW = (255, 255, 0)
# RED = (255, 0, 0)
RED = (255, 0, 0)
# PINK = (255, 184, 255)
PINK = (242, 101, 34)
# CYAN = (0, 255, 255)
CYAN = (0, 236, 255)
# ORANGE = (255, 184, 82)
ORANGE = (255, 64, 0)
# BLUE = (33, 33, 255)
BLUE = (0, 194, 222)
WHITE = (255, 255, 255)

# Ghost colors
GHOST_COLORS = [RED, PINK, CYAN, ORANGE]

# Maze layout
MAZE_TEMPLATE = [
    "############################",
    "#............##............#",
    "#.####.#####.##.#####.####.#",
    "#.####.#####.##.#####.####.#",
    "#.####.#####.##.#####.####.#",
    "#..........................#",
    "#.####.##.########.##.####.#",
    "#.####.##.########.##.####.#",
    "#......##....##....##......#",
    "######.##### ## #####.######",
    "######.##### ## #####.######",
    "######.##          ##.######",
    "######.## ###--### ##.######",
    "######.## #      # ##.######",
    "          #      #          ",
    "######.## #      # ##.######",
    "######.## ######## ##.######",
    "######.##          ##.######",
    "######.## ######## ##.######",
    "######.## ######## ##.######",
    "#............##............#",
    "#.####.#####.##.#####.####.#",
    "#.####.#####.##.#####.####.#",
    "#...##................##...#",
    "### ##.## ######## ##.## ###",
    "### ##.## ######## ##.## ###",
    "#......##....##....##......#",
    "#.##########.##.##########.#",
    "#.##########.##.##########.#",
    "#..........................#",
    "############################"
]

class PacMan:
    def __init__(self):
        self.x = 14          # column
        self.y = 23          # row

        # --- sub-tile pixel coordinates (used for smooth motion) ---
        self.pix_x = self.x * CELL_SIZE
        self.pix_y = self.y * CELL_SIZE

        self.direction = (0, 0)          # (dx, dy) set from keystrokes
        self.speed     = PAC_SPEED       # pixels / frame

        # animation & state
        self.animation_frame   = 0
        self.animation_counter = 0
        self.animation_speed   = 0.4      # frames per mouth-phase
        self.lives = 1

        self.mouth_phases = [
            (45, 315), (30, 330), (15, 345),
            (0, 360),  (15, 345), (30, 330)
        ]
        self.waka_toggle = False

    def move(self):
        new_x = self.x + self.direction[0]
        new_y = self.y + self.direction[1]

        if new_y == 14:
            if new_x < 0 and maze[14][COLS-1] != '#':
                self.x = COLS-1
                return
            if new_x >= COLS and maze[14][0] != '#':
                self.x = 0
                return

        if 0 <= new_x < COLS and 0 <= new_y < ROWS:
            if maze[new_y][new_x] != '#':
                self.x, self.y = new_x, new_y

    def update_animation(self):
        if self.direction != (0, 0):
            self.animation_counter += 1
            if self.animation_counter >= self.animation_speed * FPS:
                self.animation_frame = (self.animation_frame + 1) % len(self.mouth_phases)
                self.animation_counter = 0

    def get_mouth_angles(self):
        base = self.mouth_phases[self.animation_frame]
        if self.direction == (1,0): return base
        if self.direction == (-1,0): return (180+base[0], 180+base[1])
        if self.direction == (0,1): return (90+base[0], 90+base[1])
        if self.direction == (0,-1): return (270+base[0], 270+base[1])
        return (45, 315)

class Ghost:
    def __init__(self, x, y, color, strategy="astar"):
        self.x, self.y = x, y
        self.color = self.original_color = color
        self.directions = [(0,1),(0,-1),(1,0),(-1,0)]
        self.direction = random.choice(self.directions)
        self.strategy = strategy 

    def move(self, pacman_pos=None):
        if pacman_pos is None:
            # Default random/frightened logic
            new_x = self.x + self.direction[0]
            new_y = self.y + self.direction[1]

            if new_y == 14:
                if new_x < 0 and maze[14][COLS - 1] != '#':
                    self.x = COLS - 1
                    return
                if new_x >= COLS and maze[14][0] != '#':
                    self.x = 0
                    return

            if 0 <= new_x < COLS and 0 <= new_y < ROWS and maze[new_y][new_x] != '#':
                self.x, self.y = new_x, new_y
            else:
                valid = []
                for dx, dy in self.directions:
                    nx, ny = self.x + dx, self.y + dy
                    if ny == 14 and (nx < 0 or nx >= COLS):
                        valid.append((dx, dy))
                    elif 0 <= nx < COLS and 0 <= ny < ROWS and maze[ny][nx] != '#':
                        valid.append((dx, dy))
                if valid:
                    self.direction = random.choice(valid)
            return

        # 🔍 Choose pathfinding algorithm based on self.strategy
        path_func = {
            "astar": astar,
            "bfs": bfs,
            "greedy": greedy,
            "random": random_walk
        }.get(self.strategy, astar)

        path = path_func(maze, (self.x, self.y), pacman_pos)

        if len(path) > 0:
            next_step = path[0]
            self.direction = (next_step[0] - self.x, next_step[1] - self.y)
            self.x, self.y = next_step



def draw_maze(screen):
    for y in range(ROWS):
        for x in range(COLS):
            if maze[y][x] == '#':
                pygame.draw.rect(screen, BLUE, (x*CELL_SIZE, y*CELL_SIZE+40, CELL_SIZE, CELL_SIZE))
            elif maze[y][x] == '.':
                pygame.draw.circle(screen, WHITE, (x*CELL_SIZE+10, y*CELL_SIZE+40+10), 2)
            elif maze[y][x] == 'O':
                pygame.draw.circle(screen, WHITE, (x*CELL_SIZE+10, y*CELL_SIZE+40+10), 6)

def draw_pacman(screen, pacman):
    cx = pacman.x*CELL_SIZE + CELL_SIZE//2
    cy = pacman.y*CELL_SIZE + 40 + CELL_SIZE//2
    r = CELL_SIZE//2
    start, end = pacman.get_mouth_angles()
    start %= 360
    end %= 360
    
    pygame.draw.circle(screen, YELLOW, (cx, cy), r)
    if start != end:
        points = [(cx, cy),
                  (cx + r*math.cos(math.radians(start)), cy + r*math.sin(math.radians(start))),
                  (cx + r*math.cos(math.radians(end)), cy + r*math.sin(math.radians(end)))]
        pygame.draw.polygon(screen, BLACK, points)
    
    if pacman.direction != (0,0):
        eye_angle = (start + end)/2
        ex = cx + (r//2)*math.cos(math.radians(eye_angle))
        ey = cy + (r//2)*math.sin(math.radians(eye_angle))
        pygame.draw.circle(screen, BLACK, (ex, ey), r//4)

def draw_ghost(screen, x, y, color, direction):
    body = pygame.Rect(x*CELL_SIZE, y*CELL_SIZE+40, CELL_SIZE, CELL_SIZE)
    pygame.draw.rect(screen, color, body, border_radius=10)
    
    points = []
    for i in range(7):
        px = x*CELL_SIZE + (i*CELL_SIZE)//6
        py = y*CELL_SIZE+40 + CELL_SIZE - (CELL_SIZE//6 if i%2 else 0)
        points.append((px, py))
    points += [(x*CELL_SIZE+CELL_SIZE, y*CELL_SIZE+40+CELL_SIZE),
               (x*CELL_SIZE, y*CELL_SIZE+40+CELL_SIZE)]
    pygame.draw.polygon(screen, color, points)
    
    eye_pos = []
    pupil_offset = (0, 0)
    if direction == (1,0):
        eye_pos = [(x*CELL_SIZE+7, y*CELL_SIZE+47),
                   (x*CELL_SIZE+13, y*CELL_SIZE+47)]
        pupil_offset = (3, 0)
    elif direction == (-1,0):
        eye_pos = [(x*CELL_SIZE+7, y*CELL_SIZE+47),
                   (x*CELL_SIZE+13, y*CELL_SIZE+47)]
        pupil_offset = (-3, 0)
    elif direction == (0,-1):
        eye_pos = [(x*CELL_SIZE+7, y*CELL_SIZE+43),
                   (x*CELL_SIZE+13, y*CELL_SIZE+43)]
        pupil_offset = (0, -3)
    else:
        eye_pos = [(x*CELL_SIZE+7, y*CELL_SIZE+47),
                   (x*CELL_SIZE+13, y*CELL_SIZE+47)]
        pupil_offset = (0, 3)

    for ex, ey in eye_pos:
        pygame.draw.circle(screen, WHITE, (ex, ey), 3)
        pygame.draw.circle(screen, BLACK, (ex+pupil_offset[0], ey+pupil_offset[1]), 2)

def draw_lives(screen, lives):
    for i in range(lives):
        x = 20 + i * 40
        y = HEIGHT - 30
        pygame.draw.circle(screen, YELLOW, (x, y), 12)
        start_angle = math.radians(45)
        end_angle = math.radians(315)
        points = [
            (x, y),
            (x + 12 * math.cos(start_angle), y + 12 * math.sin(start_angle)),
            (x + 12 * math.cos(end_angle), y + 12 * math.sin(end_angle))
        ]
        pygame.draw.polygon(screen, BLACK, points)

class Game:
    def __init__(self):
        self.victory_channel = pygame.mixer.Channel(0)
        self.victory_channel.set_volume(1.0)
        self.waka_channel = pygame.mixer.Channel(1)
        self.siren_channel = pygame.mixer.Channel(2)
        self.effect_channel = pygame.mixer.Channel(3)

        self.waka_toggle = False

    def play_victory(self):
        print("🔊 Playing victory sound")
        pygame.mixer.stop()
        self.victory_channel.play(sounds['victory'])

        
    def play_start(self):
        sounds['start'].play()
        
    def play_waka(self):
        if not self.waka_channel.get_busy():
            sound = sounds['waka1'] if self.waka_toggle else sounds['waka2']
            self.waka_toggle = not self.waka_toggle
            self.waka_channel.play(sound)

    def play_siren(self):
        if not self.siren_channel.get_busy():
            self.siren_channel.play(sounds['siren'], loops=-1)        
            
    def stop_siren(self):
        self.siren_channel.stop()
        
    # def play_power_pellet(self):
    #     self.effect_channel.play(sounds['power_pellet'])
        
    # def play_ghost_eaten(self):
    #     self.effect_channel.play(sounds['ghost_eaten'])
        
    def play_death(self):
        self.siren_channel.stop()
        self.waka_channel.stop()
        self.effect_channel.play(sounds['death'])


def run_game() -> bool:
    # pygame.mixer.stop()    
    global maze
    played_victory = False  

    # ------------- set up a fresh round -------------
    maze  = MAZE_TEMPLATE.copy()
    dots  = sum(row.count('.') for row in maze)
    score = 0

    game = Game()
    # game.play_start()

    screen  = pygame.display.set_mode((WIDTH, HEIGHT))
    clock   = pygame.time.Clock()
    hud_fnt = pygame.font.Font(None, 36)
    big_fnt = pygame.font.SysFont(None, 72)

    # 3-2-1 countdown
    cdown_fnt = pygame.font.Font(None, 100)
    for i in range(3, 0, -1):
        screen.fill(BLACK)
        draw_maze(screen)
        txt = cdown_fnt.render(str(i), True, YELLOW)
        screen.blit(txt, txt.get_rect(center=(WIDTH//2, HEIGHT//2)))
        pygame.display.flip()
        pygame.time.wait(1000)

    start_time = pygame.time.get_ticks()

    pacman = PacMan()
    ghosts = [
        Ghost(13,  9, RED,    "astar"),
        Ghost(14, 11, PINK,   "bfs"),
        Ghost(13,  9, CYAN,   "greedy"),
        Ghost(14, 11, ORANGE, "random"),
    ]

    running   = True
    game_over = False
    win       = False

    # -------------------------------------------------
    while running:
        # ---------- timing & input ----------
        elapsed_ms  = pygame.time.get_ticks() - start_time
        elapsed_sec = elapsed_ms // 1000

        if (elapsed_sec >= SURVIVE_TO_WIN      # crossed threshold
                and not played_victory
                and not game_over):
            win            = True              # will show “You Win!”
            game_over      = True              # end round
            played_victory = True
            game.play_victory()

        if any((pacman.x, pacman.y) == (g.x, g.y) for g in ghosts):
            pacman.lives -= 1
            if pacman.lives <= 0:
                game_over = True

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if   event.key == pygame.K_UP:    pacman.direction = (0, -1)
                elif event.key == pygame.K_DOWN:  pacman.direction = (0,  1)
                elif event.key == pygame.K_LEFT:  pacman.direction = (-1, 0)
                elif event.key == pygame.K_RIGHT: pacman.direction = (1,  0)

        # ---------- update world ----------
        pacman.update_animation()
        pacman.move()

        # eat dot
        if maze[pacman.y][pacman.x] == '.':
            maze[pacman.y] = (maze[pacman.y][:pacman.x] + ' ' +
                              maze[pacman.y][pacman.x+1:])
            dots  -= 1
            score += 10
            # game.play_waka()
            # game.play_siren()

        # move ghosts
        for g in ghosts:
            g.move((pacman.x, pacman.y))

        # collisions
        if any((pacman.x, pacman.y) == (g.x, g.y) for g in ghosts):
            pacman.lives -= 1
            if pacman.lives <= 0:
                game_over = True
            else:
                pacman.x, pacman.y = 14, 23
                pacman.direction   = (0, 0)
                for g in ghosts:
                    g.x, g.y = 13, 9   # simple reset; tweak as desired
            # game.play_death()
            pygame.time.wait(500)

        if dots == 0:
            win = True
            game_over = True

        # ---------- drawing ----------
        screen.fill(BLACK)
        draw_maze(screen)
        draw_pacman(screen, pacman)
        for g in ghosts:
            draw_ghost(screen, g.x, g.y, g.color, g.direction)

        if not game_over:           # HUD
            draw_lives(screen, pacman.lives)
            score_s = hud_fnt.render(f"Score: {score}", True, WHITE)
            time_s  = hud_fnt.render(f"Time: {elapsed_sec}s", True, WHITE)
            screen.blit(score_s, (WIDTH//2 - score_s.get_width()//2, 10))
            screen.blit(time_s,  (10, 10))
        else:                       # overlay
            overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 64, 180))
            screen.blit(overlay, (0, 0))

            small_fnt = hud_fnt
            lines = [
                (big_fnt,   "You Win!" if win else "Game Over!", YELLOW),
                (small_fnt, f"Final Score: {score}", WHITE),
                (small_fnt, f"Time Survived: {elapsed_sec}s", WHITE),
                (small_fnt, "Press ENTER to play again", WHITE),
            ]
            total_h = sum(f.get_height() for f, *_ in lines) + 15 * (len(lines) - 1)
            y = HEIGHT // 2 - total_h // 2
            for fnt, txt, col in lines:
                surf = fnt.render(txt, True, col)
                rect = surf.get_rect(center=(WIDTH // 2, y + surf.get_height() // 2))
                screen.blit(surf, rect)
                y += surf.get_height() + 15

        pygame.display.flip()
        clock.tick(FPS)

        # ---------- restart handling ----------
        if game_over:
            waiting = True
            while waiting:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return False
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                        pygame.mixer.stop()      # ← move the stop here
                        return True              # start a new round
                clock.tick(15)
    return True

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Pac-Man")
    clock = pygame.time.Clock()
    
    # Main game loop that handles start screen and game sessions
    while True:
        # Show start screen
        if not start_screen(screen, clock):
            break
        
        # Run game session
        if not run_game():
            break
    
    pygame.quit()

if __name__ == "__main__":
    main()