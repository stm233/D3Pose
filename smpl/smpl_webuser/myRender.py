import pygame
from PIL.ImageChops import screen
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLUT import *
import pyrr
from OpenGL.GLU import gluPerspective  # Import gluPerspective
from moviepy.editor import ImageSequenceClip


class OBJ:
    def __init__(self, filename):
        self.vertices = []
        self.faces = []
        self.read_obj(filename)

    def read_obj(self, filename):
        with open(filename, 'r') as file:
            for line in file:
                if line.startswith('v '):
                    self.vertices.append(list(map(float, line.split()[1:])))
                elif line.startswith('f '):
                    self.faces.append([int(vertex.split('/')[0]) for vertex in line.split()[1:]])


def load_obj(filename):
    return OBJ(filename)


def load_multiple_objs(directory):
    objs = []
    for filename in sorted(os.listdir(directory)):
        if filename.endswith('.obj'):
            path = os.path.join(directory, filename)
            objs.append(load_obj(path))
    return objs


# Function to initialize the OpenGL environment
def init():
    glClearColor(0.0, 0.0, 0.0, 0.0)
    glMatrixMode(GL_PROJECTION)
    gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
    glTranslatef(0.0, 0.0, -5)


# Function to draw the loaded 3D object
def draw_obj(obj):
    glBegin(GL_TRIANGLES)
    for face in obj.faces:
        for vertex in face:
            glVertex3fv(obj.vertices[vertex - 1])
    glEnd()


# Main function
def main():
    pygame.init()
    global display
    display = (800, 600)
    screen = pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    init()

    objs = load_multiple_objs('./hello_world')
    frame_count = 0
    frames = []  # To store frames for video
    running = True
    obj_index = 0

    while running and obj_index < len(objs):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Update the object to be rendered
        draw_obj(objs[obj_index])

        # Capture the frame
        filename = f'frame_{frame_count}.png'
        pygame.image.save(screen, filename)
        frames.append(filename)
        frame_count += 1

        pygame.display.flip()
        pygame.time.wait(10)

        # Update obj_index to switch to the next model
        obj_index += 1

    # Compile frames into a video
    clip = ImageSequenceClip(frames, fps=20)
    clip.write_videofile('output_video.mp4', codec='mpeg4')

    # Clean up: remove frame images
    for filename in frames:
        os.remove(filename)

    # Load the 3D object using the objloader
    # obj = load_obj('hello_world/frame_22.obj')

    #
    # while True:
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             pygame.quit()
    #             quit()
    #
    #     glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    #
    #     # Handle mouse input for changing the viewing angle
    #     pressed = pygame.mouse.get_pressed()
    #     if pressed[0]:
    #         x, y = pygame.mouse.get_rel()
    #         glRotatef(pyrr.Vector3([y, x, 0.0]).length, y, x, 0.0)
    #
    #     draw_obj(obj)
    #
    #     pygame.display.flip()
    #     pygame.time.wait(10)


if __name__ == "__main__":
    main()
